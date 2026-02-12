using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Linq;
using SurpassAlpha.MC;
using SurpassAlpha.MC.Interfaces;
using SurpassAlpha.RouseModel;

namespace RouseValidationApp
{
    class Program
    {
        // Default simulation parameters
        static SegmentModeEnum SegmentMode = SegmentModeEnum.Coarse;
        static int ChainLength = 100;
        static int NumChains = 20;
        static int Seed = 42;
        static int EquilibrationSweeps = 100;
        static int ProductionSweeps = 200;
        static int ResiduesPerSegment = 0; // auto from segment mode
        static int MoveSize = 0; // 0 = auto (one move per segment per sweep)
        static double MaxDisplacement = 0.5;
        static double MaxAngle = Math.PI; // π radians (180°) for optimal 30-50% acceptance
        static double BoxSizeOverride = 0; // 0 = auto-compute
        static string OutputDir = "production_output";
        static bool BatchMode = false;
        static int[] BatchChainLengths = new int[] { 25, 50, 75, 100 };

        // Physical constants
        const double Sigma = 3.8;
        const double Temperature = 300.0;

        static void Main(string[] args)
        {
            ParseArguments(args);

            double boxSize;
            if (BoxSizeOverride > 0)
            {
                boxSize = BoxSizeOverride;
            }
            else
            {
                // With MIC-corrected movers, chains can safely cross PBC boundaries.
                // Box size controls density and thus acceptance rate (target: 30-50%).
                // Empirically calibrated: volume fraction phi ~ 0.035 gives ~40% acceptance
                // for athermal hinge MC with max_angle = pi.
                double targetPhi = 0.035;
                double sigma3 = Sigma * Sigma * Sigma;
                boxSize = Math.Pow(NumChains * ChainLength * sigma3 / targetPhi, 1.0 / 3.0);
                // Minimum: 3 * R_rms to avoid chain self-interaction through PBC
                double rmsR = Sigma * Math.Pow(ChainLength, 0.588);
                boxSize = Math.Max(boxSize, 3.0 * rmsR);
            }
            if (ResiduesPerSegment == 0)
                ResiduesPerSegment = (SegmentMode == SegmentModeEnum.Coarse) ? 20 : 10;
            if (ResiduesPerSegment >= ChainLength)
                ResiduesPerSegment = Math.Max(2, ChainLength / 3);

            // MoveSize = total segments for a proper MC sweep (one attempt per segment)
            int totalSegments = NumChains * (int)Math.Ceiling((double)ChainLength / ResiduesPerSegment);
            if (MoveSize <= 0)
                MoveSize = totalSegments;
            else
                MoveSize = Math.Min(MoveSize, totalSegments);

            Console.WriteLine("================================================================");
            Console.WriteLine("  ROUSE POLYMER SCALING VALIDATION");
            Console.WriteLine("================================================================");
            Console.WriteLine();

            PrintParameters(boxSize);

            try
            {
                if (BatchMode)
                {
                    RunBatch();
                }
                else
                {
                    RunValidation(boxSize);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(string.Format("[FATAL] {0}", ex.Message));
                Console.WriteLine(ex.StackTrace);
                Environment.Exit(1);
            }
        }

        static void RunValidation(double boxSize)
        {
            // Create factory
            RouseValidationSegmentedFactory factory = new RouseValidationSegmentedFactory(
                ChainLength, NumChains, MaxDisplacement, boxSize, Seed, SegmentMode, ResiduesPerSegment, MaxAngle);

            // Execution mode diagnostics
            Console.WriteLine("=== EXECUTION DIAGNOSTICS ===");
            bool gpuAvailable = GpuEnergyFactory.IsGpuAvailable();
            Console.WriteLine(string.Format("GPU OpenCL Available: {0}", gpuAvailable));
            Console.WriteLine(string.Format("CPU Threads:          {0}", Environment.ProcessorCount));
            if (!gpuAvailable)
            {
                Console.WriteLine("Execution Mode:       CPU (serial fallback - OpenCL unavailable)");
                Console.WriteLine();
                Console.WriteLine("--- OpenCL Diagnostics ---");
            }
            else
            {
                Console.WriteLine("Execution Mode:       GPU (OpenCL)");
                Console.WriteLine();
                Console.WriteLine("--- GPU Device Info ---");
            }
            Console.WriteLine(GpuEnergyFactory.GetDiagnostics());
            Console.WriteLine();

            // Create Rouse observers
            MultiChainCenterOfMassMsdObserver cmMsdObserver = new MultiChainCenterOfMassMsdObserver();
            MultiChainMiddleSegmentMsdObserver g1Observer = new MultiChainMiddleSegmentMsdObserver();
            MultiChainEndToEndAutocorrelationObserver autocorrObserver = new MultiChainEndToEndAutocorrelationObserver();

            // Create configuration
            SegmentedMultiStepMCConfiguration config = new SegmentedMultiStepMCConfiguration();
            config.Temperature = Temperature;
            config.NumberOfSweeps = ProductionSweeps;
            config.EquilibrationSweeps = EquilibrationSweeps;
            config.OutputInterval = 1; // fire SweepCompleted every sweep for observers
            config.RandomSeed = Seed;
            config.ProgressLogInterval = Math.Max(1, ProductionSweeps / 10);
            config.TrackEnergyHistory = false;
            config.MoveSize = MoveSize;

            // Progress logging interval
            int progressInterval = Math.Max(10, ProductionSweeps / 10);

            // Create and initialize simulation (creates system, energy computer, etc.)
            Console.WriteLine("=== INITIALIZING ===");
            Stopwatch initWatch = Stopwatch.StartNew();
            SegmentedMultiStepMC simulation = new SegmentedMultiStepMC(factory, config, null);
            simulation.Initialize();
            initWatch.Stop();
            Console.WriteLine(string.Format("Initialization complete ({0:F1}s)", initWatch.Elapsed.TotalSeconds));
            Console.WriteLine();

            // Timing for ETA (separate stopwatches for each phase)
            Stopwatch eqWatch = new Stopwatch();
            Stopwatch prodWatch = new Stopwatch();

            // Observer baselines must be captured AFTER equilibration (at start of
            // production), not before. Otherwise MSD measures displacement from the
            // pre-equilibration snapshot, burying the diffusive signal in noise, and
            // the autocorrelation starts near zero instead of 1.
            bool baselinesCaptured = false;
            bool eqPhaseEnded = false;

            // Hook sweep events for observer updates (production only)
            simulation.SweepCompleted += (sender, state) =>
            {
                int sweep = state.CurrentSweep;
                ISegmentedSystem seg = simulation.SegmentedSystem;
                ISimulationSystem sys = seg as ISimulationSystem;
                if (sys != null)
                {
                    if (!baselinesCaptured)
                    {
                        // Initialize observers with post-equilibration coordinates
                        cmMsdObserver.OnSimulationStart(sys);
                        g1Observer.OnSimulationStart(sys);
                        autocorrObserver.OnSimulationStart(sys);
                        cmMsdObserver.OnSweep(sys, 0);
                        g1Observer.OnSweep(sys, 0);
                        autocorrObserver.OnSweep(sys, 0);
                        baselinesCaptured = true;
                    }
                    cmMsdObserver.OnSweep(sys, sweep);
                    g1Observer.OnSweep(sys, sweep);
                    autocorrObserver.OnSweep(sys, sweep);
                }
            };

            // Hook every-sweep event for progress bar rendering
            simulation.SweepPerformed += (absoluteSweep, totalSweeps, isProduction) =>
            {
                if (!isProduction)
                {
                    // Equilibration phase
                    if (!eqWatch.IsRunning)
                        eqWatch.Start();

                    int eqSweep = absoluteSweep;
                    RenderProgressBar("Equilibration", eqSweep, EquilibrationSweeps,
                        eqWatch, simulation.CurrentEnergy, simulation.AcceptanceRatio, false);
                }
                else
                {
                    // Transition: finish equilibration bar, start production
                    if (!eqPhaseEnded)
                    {
                        if (EquilibrationSweeps > 0)
                        {
                            eqWatch.Stop();
                            RenderProgressBar("Equilibration", EquilibrationSweeps, EquilibrationSweeps,
                                eqWatch, 0, 0, false);
                            Console.WriteLine();
                            Console.WriteLine(string.Format("Equilibration done in {0}",
                                FormatElapsed(eqWatch.Elapsed.TotalSeconds)));
                            Console.WriteLine();
                        }
                        eqPhaseEnded = true;
                        prodWatch.Start();
                    }

                    int prodSweep = absoluteSweep - EquilibrationSweeps;
                    RenderProgressBar("Production   ", prodSweep, ProductionSweeps,
                        prodWatch, simulation.CurrentEnergy, simulation.AcceptanceRatio, true);
                }
            };

            Console.WriteLine("=== RUNNING SIMULATION ===");
            Console.WriteLine(string.Format("Segment mode: {0}", SegmentMode));
            Console.WriteLine(string.Format("Chain length: {0}", ChainLength));
            Console.WriteLine(string.Format("Chains: {0}", NumChains));
            Console.WriteLine(string.Format("Total sweeps: {0} eq + {1} prod = {2}",
                EquilibrationSweeps, ProductionSweeps, EquilibrationSweeps + ProductionSweeps));
            Console.WriteLine();

            // Run simulation
            Stopwatch totalWatch = Stopwatch.StartNew();
            IMCState finalState = simulation.Run();
            totalWatch.Stop();

            // Finish the production progress bar
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine(string.Format("Total wall time: {0}", FormatElapsed(totalWatch.Elapsed.TotalSeconds)));

            // Report results
            Console.WriteLine();
            Console.WriteLine("=== SIMULATION COMPLETE ===");
            Console.WriteLine(string.Format("Final tracked energy: {0:F4}", finalState.Energy));

            // Recompute energy from scratch to verify tracking
            ISegmentedNonBondedEnergyComputer verifyEnergy = factory.CreateEnergyComputer(simulation.SegmentedSystem);
            double recomputedEnergy = verifyEnergy.GetTotalPotential();
            Console.WriteLine(string.Format("Recomputed energy:    {0:F4}", recomputedEnergy));
            Console.WriteLine(string.Format("Energy drift:         {0:E4}", finalState.Energy - recomputedEnergy));

            Console.WriteLine(string.Format("Acceptance ratio: {0:P2}", finalState.AcceptanceRatio));
            Console.WriteLine(string.Format("Lowest energy: {0:F4}", finalState.LowestEnergy));

            // Validate acceptance rate
            double finalAcceptRate = finalState.AcceptanceRatio;
            if (finalAcceptRate >= 0.30 && finalAcceptRate <= 0.50)
                Console.WriteLine("[PASS] Acceptance rate in 30-50% range");
            else
                Console.WriteLine(string.Format("[WARN] Acceptance rate {0:P2} outside 30-50% target", finalAcceptRate));

            // Compute static properties
            ComputeStaticProperties(simulation.SegmentedSystem);

            // Write output files
            WriteOutputFiles(cmMsdObserver, g1Observer, autocorrObserver, simulation.SegmentedSystem);

            // Write summary line for batch processing
            Console.WriteLine();
            Console.WriteLine(string.Format("=== SUMMARY: alg={0} N={1} chains={2} seed={3} accept={4:P2} ===",
                SegmentMode, ChainLength, NumChains, Seed, finalState.AcceptanceRatio));
            Console.WriteLine("Output written to: " + Path.GetFullPath(OutputDir));
        }

        static void RunBatch()
        {
            Console.WriteLine("================================================================");
            Console.WriteLine("  BATCH MODE: Scaling Law Validation (Figs 1-4)");
            Console.WriteLine("================================================================");
            Console.WriteLine();

            string physDir = Path.Combine(OutputDir, "physical_unit");
            Directory.CreateDirectory(physDir);

            string scalingFile = Path.Combine(physDir, "fig1_scaling_R2_Rg2.tsv");
            using (StreamWriter scalingWriter = new StreamWriter(scalingFile))
            {
                scalingWriter.WriteLine("N\tR2\tRg2");

                int originalChainLength = ChainLength;

                foreach (int n in BatchChainLengths)
                {
                    Console.WriteLine(string.Format("=== BATCH: N={0} ===", n));
                    ChainLength = n;

                    // Recompute segment size for this chain length
                    ResiduesPerSegment = (SegmentMode == SegmentModeEnum.Coarse) ? 20 : 10;
                    if (ResiduesPerSegment >= ChainLength)
                        ResiduesPerSegment = Math.Max(2, ChainLength / 3);
                    MoveSize = NumChains * (int)Math.Ceiling((double)ChainLength / ResiduesPerSegment);

                    // Auto box size for this chain length (same formula as Main)
                    double targetPhi = 0.035;
                    double sigma3 = Sigma * Sigma * Sigma;
                    double boxSize = Math.Pow(NumChains * ChainLength * sigma3 / targetPhi, 1.0 / 3.0);
                    double rmsR = Sigma * Math.Pow(ChainLength, 0.588);
                    boxSize = Math.Max(boxSize, 3.0 * rmsR);

                    try
                    {
                        RunValidation(boxSize);

                        // Read back the static properties file to get averages
                        string staticFile = Path.Combine(physDir,
                            string.Format("fig1_static_N{0}_s{1}.tsv", n, Seed));
                        if (File.Exists(staticFile))
                        {
                            double sumR2 = 0, sumRg2 = 0;
                            int count = 0;
                            foreach (string line in File.ReadAllLines(staticFile))
                            {
                                if (line.StartsWith("chain")) continue;
                                string[] parts = line.Split('\t');
                                if (parts.Length >= 3)
                                {
                                    sumR2 += double.Parse(parts[1], CultureInfo.InvariantCulture);
                                    sumRg2 += double.Parse(parts[2], CultureInfo.InvariantCulture);
                                    count++;
                                }
                            }
                            if (count > 0)
                            {
                                scalingWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
                                    "{0}\t{1:F6}\t{2:F6}", n, sumR2 / count, sumRg2 / count));
                                scalingWriter.Flush();
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(string.Format("[ERROR] N={0}: {1}", n, ex.Message));
                    }

                    Console.WriteLine();
                }

                ChainLength = originalChainLength;
            }

            Console.WriteLine(string.Format("Wrote scaling data: {0}", scalingFile));

            // Generate PNG plots from batch data
            GeneratePlots(physDir);
        }

        static void ComputeStaticProperties(ISegmentedSystem segSystem)
        {
            ISimulationSystem system = segSystem as ISimulationSystem;
            if (system == null) return;

            Console.WriteLine();
            Console.WriteLine("=== STATIC PROPERTIES ===");

            // NOTE: Range.End is INCLUSIVE in SimulationSystem
            int chainCount = system.GetChainsCount();
            INumberSpace ns = system.NumberSpace;

            double sumR2 = 0.0;
            double sumRg2 = 0.0;

            for (int c = 0; c < chainCount; c++)
            {
                Range chainRange = system.GetChainRange(c);
                int first = chainRange.Begin;
                int last = chainRange.End;   // End is INCLUSIVE
                int atomCount = last - first + 1;

                // Cumulative MIC unwrapping
                double[] ux = new double[atomCount];
                double[] uy = new double[atomCount];
                double[] uz = new double[atomCount];
                IVec3DoubleIdx firstPos = system.GetVec3Double(first);
                ux[0] = firstPos.X;
                uy[0] = firstPos.Y;
                uz[0] = firstPos.Z;
                for (int i = 1; i < atomCount; i++)
                {
                    IVec3DoubleIdx prev = system.GetVec3Double(first + i - 1);
                    IVec3DoubleIdx curr = system.GetVec3Double(first + i);
                    ux[i] = ux[i - 1] + ns.GetMICDistanceDouble(prev.X, curr.X);
                    uy[i] = uy[i - 1] + ns.GetMICDistanceDouble(prev.Y, curr.Y);
                    uz[i] = uz[i - 1] + ns.GetMICDistanceDouble(prev.Z, curr.Z);
                }

                // CM from unwrapped
                double cmX = 0, cmY = 0, cmZ = 0;
                for (int i = 0; i < atomCount; i++)
                { cmX += ux[i]; cmY += uy[i]; cmZ += uz[i]; }
                cmX /= atomCount; cmY /= atomCount; cmZ /= atomCount;

                // Rg²
                double rg2 = 0.0;
                for (int i = 0; i < atomCount; i++)
                {
                    double dx = ux[i] - cmX;
                    double dy = uy[i] - cmY;
                    double dz = uz[i] - cmZ;
                    rg2 += dx * dx + dy * dy + dz * dz;
                }
                rg2 /= atomCount;
                sumRg2 += rg2;

                // R²
                double reX = ux[atomCount - 1] - ux[0];
                double reY = uy[atomCount - 1] - uy[0];
                double reZ = uz[atomCount - 1] - uz[0];
                double r2 = reX * reX + reY * reY + reZ * reZ;
                sumR2 += r2;

                // Per-chain output for first few chains
                if (c < 3)
                {
                    Console.WriteLine(string.Format("  Chain {0}: R2={1:F4}, Rg2={2:F4}, ratio={3:F4}",
                        c, r2, rg2, rg2 > 0 ? r2 / rg2 : 0));
                }
            }

            double avgR2 = sumR2 / chainCount;
            double avgRg2 = sumRg2 / chainCount;
            double ratio = avgRg2 > 0 ? avgR2 / avgRg2 : 0;

            Console.WriteLine(string.Format("<R^2>   = {0:F4}", avgR2));
            Console.WriteLine(string.Format("<Rg^2>  = {0:F4}", avgRg2));
            Console.WriteLine(string.Format("R^2/Rg^2 = {0:F4} (theory ~ 6.0 for SAW)", ratio));
        }

        static void WriteOutputFiles(
            MultiChainCenterOfMassMsdObserver cmMsd,
            MultiChainMiddleSegmentMsdObserver g1Msd,
            MultiChainEndToEndAutocorrelationObserver autocorr,
            ISegmentedSystem segSystem)
        {
            string physDir = Path.Combine(OutputDir, "physical_unit");
            Directory.CreateDirectory(physDir);

            string algTag = SegmentMode.ToString().ToLowerInvariant();

            // Write CM MSD (diffusion data for Fig 3)
            if (cmMsd.HasData)
            {
                string fileName = string.Format("fig3_{0}_diffusion_N{1}_s{2}.tsv", algTag, ChainLength, Seed);
                string filePath = Path.Combine(physDir, fileName);
                WriteTsv(filePath, "sweep\tg_CM", cmMsd.GetMsdCurve());
                Console.WriteLine(string.Format("Wrote: {0}", filePath));

                // Estimate diffusion coefficient
                try
                {
                    DiffusionCoefficientEstimator estimator = new DiffusionCoefficientEstimator();
                    double D = estimator.Estimate(cmMsd.GetMsdCurve());
                    Console.WriteLine(string.Format("Diffusion coefficient D = {0:E4}", D));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(string.Format("Could not estimate D: {0}", ex.Message));
                }
            }

            // Write middle bead MSD (Fig 2 data)
            if (g1Msd.HasData)
            {
                string fileName = string.Format("fig2_{0}_msd_N{1}_s{2}.tsv", algTag, ChainLength, Seed);
                string filePath = Path.Combine(physDir, fileName);
                WriteTsv(filePath, "sweep\tg1", g1Msd.GetMsdCurve());
                Console.WriteLine(string.Format("Wrote: {0}", filePath));
            }

            // Write autocorrelation (Fig 4 data for relaxation time)
            if (autocorr.HasData)
            {
                string fileName = string.Format("fig4_{0}_autocorr_N{1}_s{2}.tsv", algTag, ChainLength, Seed);
                string filePath = Path.Combine(physDir, fileName);
                WriteTsv(filePath, "sweep\tgR", autocorr.GetAutocorrelation());
                Console.WriteLine(string.Format("Wrote: {0}", filePath));

                // Estimate relaxation time
                try
                {
                    RelaxationTimeEstimator relaxEstimator = new RelaxationTimeEstimator();
                    double tauR = relaxEstimator.EstimateTauR(autocorr.GetAutocorrelation());
                    Console.WriteLine(string.Format("Relaxation time tau_R = {0:F2}", tauR));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(string.Format("Could not estimate tau_R: {0}", ex.Message));
                }
            }

            // Write static properties (Fig 1 data)
            ISimulationSystem system = segSystem as ISimulationSystem;
            if (system != null)
            {
                WriteStaticPropertiesTsv(physDir, algTag, system);
            }
        }

        static void WriteStaticPropertiesTsv(string physDir, string algTag, ISimulationSystem system)
        {
            int chainCount = system.GetChainsCount();
            INumberSpace ns = system.NumberSpace;
            string fileName = string.Format("fig1_static_N{0}_s{1}.tsv", ChainLength, Seed);
            string filePath = Path.Combine(physDir, fileName);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("chain\tR2\tRg2");

                for (int c = 0; c < chainCount; c++)
                {
                    Range chainRange = system.GetChainRange(c);
                    int first = chainRange.Begin;
                    int last = chainRange.End;       // INCLUSIVE
                    int atomCount = last - first + 1;

                    // Cumulative MIC unwrapping (same as ComputeStaticProperties)
                    double[] ux = new double[atomCount];
                    double[] uy = new double[atomCount];
                    double[] uz = new double[atomCount];
                    IVec3DoubleIdx fp = system.GetVec3Double(first);
                    ux[0] = fp.X; uy[0] = fp.Y; uz[0] = fp.Z;
                    for (int i = 1; i < atomCount; i++)
                    {
                        IVec3DoubleIdx prev = system.GetVec3Double(first + i - 1);
                        IVec3DoubleIdx curr = system.GetVec3Double(first + i);
                        ux[i] = ux[i - 1] + ns.GetMICDistanceDouble(prev.X, curr.X);
                        uy[i] = uy[i - 1] + ns.GetMICDistanceDouble(prev.Y, curr.Y);
                        uz[i] = uz[i - 1] + ns.GetMICDistanceDouble(prev.Z, curr.Z);
                    }

                    // R² from unwrapped endpoints
                    double reX = ux[atomCount - 1] - ux[0];
                    double reY = uy[atomCount - 1] - uy[0];
                    double reZ = uz[atomCount - 1] - uz[0];
                    double r2 = reX * reX + reY * reY + reZ * reZ;

                    // CM from unwrapped
                    double cmX = 0, cmY = 0, cmZ = 0;
                    for (int i = 0; i < atomCount; i++)
                    { cmX += ux[i]; cmY += uy[i]; cmZ += uz[i]; }
                    cmX /= atomCount; cmY /= atomCount; cmZ /= atomCount;

                    // Rg² from unwrapped
                    double rg2 = 0.0;
                    for (int i = 0; i < atomCount; i++)
                    {
                        double dx = ux[i] - cmX;
                        double dy = uy[i] - cmY;
                        double dz = uz[i] - cmZ;
                        rg2 += dx * dx + dy * dy + dz * dz;
                    }
                    rg2 /= atomCount;

                    writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0}\t{1:F6}\t{2:F6}", c, r2, rg2));
                }
            }

            Console.WriteLine(string.Format("Wrote: {0}", filePath));
        }

        static void WriteTsv(string filePath, string header, IReadOnlyDictionary<int, double> data)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(header);
                foreach (KeyValuePair<int, double> kv in data)
                {
                    writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0}\t{1:E8}", kv.Key, kv.Value));
                }
            }
        }

        static void GeneratePlots(string physDir)
        {
            try
            {
                // Fig 1: R^2 and Rg^2 vs N (log-log)
                string scalingFile = Path.Combine(physDir, "fig1_scaling_R2_Rg2.tsv");
                if (File.Exists(scalingFile))
                {
                    var ns = new List<double>();
                    var r2s = new List<double>();
                    var rg2s = new List<double>();
                    foreach (string line in File.ReadAllLines(scalingFile))
                    {
                        if (line.StartsWith("N")) continue;
                        string[] parts = line.Split('\t');
                        if (parts.Length >= 3)
                        {
                            ns.Add(double.Parse(parts[0], CultureInfo.InvariantCulture));
                            r2s.Add(double.Parse(parts[1], CultureInfo.InvariantCulture));
                            rg2s.Add(double.Parse(parts[2], CultureInfo.InvariantCulture));
                        }
                    }
                    if (ns.Count >= 2)
                    {
                        DrawLogLogPlot(
                            Path.Combine(physDir, "fig1_R2_Rg2_vs_N.png"),
                            "Fig 1: R^2 and Rg^2 vs N",
                            "N", "Value",
                            ns.ToArray(), r2s.ToArray(), "R^2",
                            ns.ToArray(), rg2s.ToArray(), "Rg^2");
                        Console.WriteLine("Wrote: " + Path.Combine(physDir, "fig1_R2_Rg2_vs_N.png"));
                    }
                }

                // Fig 2: MSD g1 vs t (if single-N data exists)
                string algTag = SegmentMode.ToString().ToLowerInvariant();
                foreach (int n in BatchChainLengths)
                {
                    string msdFile = Path.Combine(physDir,
                        string.Format("fig2_{0}_msd_N{1}_s{2}.tsv", algTag, n, Seed));
                    if (File.Exists(msdFile))
                    {
                        var sweeps = new List<double>();
                        var vals = new List<double>();
                        foreach (string line in File.ReadAllLines(msdFile))
                        {
                            if (line.StartsWith("sweep")) continue;
                            string[] parts = line.Split('\t');
                            if (parts.Length >= 2)
                            {
                                double sw = double.Parse(parts[0], CultureInfo.InvariantCulture);
                                double val = double.Parse(parts[1], CultureInfo.InvariantCulture);
                                if (sw > 0 && val > 0) { sweeps.Add(sw); vals.Add(val); }
                            }
                        }
                        if (sweeps.Count >= 2)
                        {
                            DrawLogLogPlot(
                                Path.Combine(physDir, string.Format("fig2_msd_g1_N{0}.png", n)),
                                string.Format("Fig 2: g1(t) N={0}", n),
                                "sweep", "g1",
                                sweeps.ToArray(), vals.ToArray(), "g1(t)",
                                null, null, null);
                        }
                    }
                }

                // Fig 3: D vs N (from diffusion files)
                {
                    var dNs = new List<double>();
                    var dVals = new List<double>();
                    foreach (int n in BatchChainLengths)
                    {
                        string diffFile = Path.Combine(physDir,
                            string.Format("fig3_{0}_diffusion_N{1}_s{2}.tsv", algTag, n, Seed));
                        if (File.Exists(diffFile))
                        {
                            var sweeps = new List<double>();
                            var msdVals = new List<double>();
                            foreach (string line in File.ReadAllLines(diffFile))
                            {
                                if (line.StartsWith("sweep")) continue;
                                string[] parts = line.Split('\t');
                                if (parts.Length >= 2)
                                {
                                    double sw = double.Parse(parts[0], CultureInfo.InvariantCulture);
                                    double val = double.Parse(parts[1], CultureInfo.InvariantCulture);
                                    if (sw > 0 && val > 0) { sweeps.Add(sw); msdVals.Add(val); }
                                }
                            }
                            if (sweeps.Count >= 2)
                            {
                                // D = MSD / (6 * t) using last quarter of data
                                int start = sweeps.Count * 3 / 4;
                                double avgD = 0;
                                int cnt = 0;
                                for (int i = start; i < sweeps.Count; i++)
                                {
                                    avgD += msdVals[i] / (6.0 * sweeps[i]);
                                    cnt++;
                                }
                                if (cnt > 0)
                                {
                                    dNs.Add(n);
                                    dVals.Add(avgD / cnt);
                                }
                            }
                        }
                    }
                    if (dNs.Count >= 2)
                    {
                        DrawLogLogPlot(
                            Path.Combine(physDir, "fig3_diffusion_D_vs_N.png"),
                            "Fig 3: D vs N",
                            "N", "D",
                            dNs.ToArray(), dVals.ToArray(), "D",
                            null, null, null);
                        Console.WriteLine("Wrote: " + Path.Combine(physDir, "fig3_diffusion_D_vs_N.png"));
                    }
                }

                // Fig 4: tau_R vs N (from autocorrelation files)
                {
                    var tNs = new List<double>();
                    var tVals = new List<double>();
                    foreach (int n in BatchChainLengths)
                    {
                        string acFile = Path.Combine(physDir,
                            string.Format("fig4_{0}_autocorr_N{1}_s{2}.tsv", algTag, n, Seed));
                        if (File.Exists(acFile))
                        {
                            var sweeps = new List<double>();
                            var acVals = new List<double>();
                            foreach (string line in File.ReadAllLines(acFile))
                            {
                                if (line.StartsWith("sweep")) continue;
                                string[] parts = line.Split('\t');
                                if (parts.Length >= 2)
                                {
                                    double sw = double.Parse(parts[0], CultureInfo.InvariantCulture);
                                    double val = double.Parse(parts[1], CultureInfo.InvariantCulture);
                                    sweeps.Add(sw); acVals.Add(val);
                                }
                            }
                            // tau_R: sweep where autocorrelation drops below 1/e
                            double threshold = 1.0 / Math.E;
                            for (int i = 0; i < acVals.Count; i++)
                            {
                                if (acVals[i] <= threshold && sweeps[i] > 0)
                                {
                                    tNs.Add(n);
                                    tVals.Add(sweeps[i]);
                                    break;
                                }
                            }
                        }
                    }
                    if (tNs.Count >= 2)
                    {
                        DrawLogLogPlot(
                            Path.Combine(physDir, "fig4_relaxation_tau_vs_N.png"),
                            "Fig 4: tau_R vs N",
                            "N", "tau_R",
                            tNs.ToArray(), tVals.ToArray(), "tau_R",
                            null, null, null);
                        Console.WriteLine("Wrote: " + Path.Combine(physDir, "fig4_relaxation_tau_vs_N.png"));
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(string.Format("[WARN] Plot generation failed: {0}", ex.Message));
            }
        }

        static void DrawLogLogPlot(
            string filePath, string title, string xLabel, string yLabel,
            double[] x1, double[] y1, string label1,
            double[] x2, double[] y2, string label2)
        {
            int w = 640, h = 480;
            int margin = 60;

            using (Bitmap bmp = new Bitmap(w, h))
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.Clear(Color.White);

                // Combine all data to find axis ranges
                double minX = double.MaxValue, maxX = double.MinValue;
                double minY = double.MaxValue, maxY = double.MinValue;
                foreach (double v in x1) { if (v > 0) { minX = Math.Min(minX, v); maxX = Math.Max(maxX, v); } }
                foreach (double v in y1) { if (v > 0) { minY = Math.Min(minY, v); maxY = Math.Max(maxY, v); } }
                if (x2 != null)
                {
                    foreach (double v in x2) { if (v > 0) { minX = Math.Min(minX, v); maxX = Math.Max(maxX, v); } }
                    foreach (double v in y2) { if (v > 0) { minY = Math.Min(minY, v); maxY = Math.Max(maxY, v); } }
                }

                double logMinX = Math.Floor(Math.Log10(minX));
                double logMaxX = Math.Ceiling(Math.Log10(maxX));
                double logMinY = Math.Floor(Math.Log10(minY));
                double logMaxY = Math.Ceiling(Math.Log10(maxY));
                if (logMaxX <= logMinX) logMaxX = logMinX + 1;
                if (logMaxY <= logMinY) logMaxY = logMinY + 1;

                int plotW = w - 2 * margin;
                int plotH = h - 2 * margin;

                // Draw axes
                using (Pen axisPen = new Pen(Color.Black, 1.5f))
                using (Font font = new Font("Arial", 9))
                using (Font titleFont = new Font("Arial", 11, FontStyle.Bold))
                {
                    g.DrawRectangle(axisPen, margin, margin, plotW, plotH);

                    // Title
                    SizeF titleSize = g.MeasureString(title, titleFont);
                    g.DrawString(title, titleFont, Brushes.Black,
                        (w - titleSize.Width) / 2, 5);

                    // Axis labels
                    g.DrawString(xLabel, font, Brushes.Black, w / 2 - 10, h - 20);
                    g.TranslateTransform(15, h / 2 + 10);
                    g.RotateTransform(-90);
                    g.DrawString(yLabel, font, Brushes.Black, 0, 0);
                    g.ResetTransform();

                    // Tick labels
                    for (double lx = logMinX; lx <= logMaxX; lx += 1.0)
                    {
                        float px = (float)(margin + plotW * (lx - logMinX) / (logMaxX - logMinX));
                        g.DrawLine(Pens.LightGray, px, margin, px, margin + plotH);
                        g.DrawString(string.Format("1e{0}", (int)lx), font, Brushes.Black, px - 10, margin + plotH + 3);
                    }
                    for (double ly = logMinY; ly <= logMaxY; ly += 1.0)
                    {
                        float py = (float)(margin + plotH - plotH * (ly - logMinY) / (logMaxY - logMinY));
                        g.DrawLine(Pens.LightGray, margin, py, margin + plotW, py);
                        g.DrawString(string.Format("1e{0}", (int)ly), font, Brushes.Black, 5, py - 6);
                    }

                    // Plot data series 1
                    DrawSeries(g, x1, y1, logMinX, logMaxX, logMinY, logMaxY,
                        margin, plotW, plotH, Color.Blue, label1, font, margin + 5, margin + 5);

                    // Plot data series 2
                    if (x2 != null && y2 != null)
                    {
                        DrawSeries(g, x2, y2, logMinX, logMaxX, logMinY, logMaxY,
                            margin, plotW, plotH, Color.Red, label2, font, margin + 5, margin + 20);
                    }

                    // Fit line for series 1 and annotate exponent
                    if (x1.Length >= 2)
                    {
                        double sumLx = 0, sumLy = 0, sumLxLy = 0, sumLx2 = 0;
                        int cnt = 0;
                        for (int i = 0; i < x1.Length; i++)
                        {
                            if (x1[i] > 0 && y1[i] > 0)
                            {
                                double lx = Math.Log10(x1[i]);
                                double ly = Math.Log10(y1[i]);
                                sumLx += lx; sumLy += ly;
                                sumLxLy += lx * ly; sumLx2 += lx * lx;
                                cnt++;
                            }
                        }
                        if (cnt >= 2)
                        {
                            double slope = (cnt * sumLxLy - sumLx * sumLy) / (cnt * sumLx2 - sumLx * sumLx);
                            string annotation = string.Format("slope = {0:F3}", slope);
                            g.DrawString(annotation, font, Brushes.DarkBlue, margin + plotW - 100, margin + 5);
                        }
                    }
                }

                bmp.Save(filePath, System.Drawing.Imaging.ImageFormat.Png);
            }
        }

        static void DrawSeries(Graphics g, double[] x, double[] y,
            double logMinX, double logMaxX, double logMinY, double logMaxY,
            int margin, int plotW, int plotH, Color color, string label,
            Font font, float legendX, float legendY)
        {
            using (Pen pen = new Pen(color, 2f))
            using (Brush brush = new SolidBrush(color))
            {
                PointF prevPt = PointF.Empty;
                bool hasPrev = false;

                for (int i = 0; i < x.Length; i++)
                {
                    if (x[i] <= 0 || y[i] <= 0) continue;
                    double lx = Math.Log10(x[i]);
                    double ly = Math.Log10(y[i]);
                    float px = (float)(margin + plotW * (lx - logMinX) / (logMaxX - logMinX));
                    float py = (float)(margin + plotH - plotH * (ly - logMinY) / (logMaxY - logMinY));

                    g.FillEllipse(brush, px - 4, py - 4, 8, 8);

                    if (hasPrev)
                        g.DrawLine(pen, prevPt, new PointF(px, py));

                    prevPt = new PointF(px, py);
                    hasPrev = true;
                }

                // Legend
                g.FillRectangle(brush, legendX, legendY, 10, 10);
                g.DrawString(label, font, brush, legendX + 14, legendY - 2);
            }
        }

        static void PrintParameters(double boxSize)
        {
            Console.WriteLine("=== PARAMETERS ===");
            Console.WriteLine(string.Format("Segment mode:       {0}", SegmentMode));
            Console.WriteLine(string.Format("Chain length:       {0}", ChainLength));
            Console.WriteLine(string.Format("Number of chains:   {0}", NumChains));
            Console.WriteLine(string.Format("Residues/segment:   {0}", ResiduesPerSegment));
            Console.WriteLine(string.Format("Max displacement:   {0:F2} A", MaxDisplacement));
            Console.WriteLine(string.Format("Max angle:          {0:F4} rad ({1:F1} deg)", MaxAngle > 0 ? MaxAngle : Math.PI / 2, (MaxAngle > 0 ? MaxAngle : Math.PI / 2) * 180.0 / Math.PI));
            Console.WriteLine(string.Format("Box size:           {0:F1} A", boxSize));
            Console.WriteLine(string.Format("Temperature:        {0:F1} K", Temperature));
            Console.WriteLine(string.Format("Equilibration:      {0} sweeps", EquilibrationSweeps));
            Console.WriteLine(string.Format("Production:         {0} sweeps", ProductionSweeps));
            Console.WriteLine(string.Format("Move size:          {0}", MoveSize));
            Console.WriteLine(string.Format("Seed:               {0}", Seed));
            Console.WriteLine(string.Format("Output directory:   {0}", OutputDir));
            Console.WriteLine();
        }

        static void RenderProgressBar(string phase, int current, int total,
            Stopwatch sw, double energy, double acceptRate, bool showEnergy)
        {
            double pct = total > 0 ? (double)current / total : 0;
            int barWidth = 40;
            int filled = (int)(pct * barWidth);
            string bar = new string('#', filled) + new string('.', barWidth - filled);

            double elapsed = sw.Elapsed.TotalSeconds;
            double sweepsPerSec = current > 0 && elapsed > 0 ? (double)current / elapsed : 0;
            int remaining = total - current;
            double etaSec = sweepsPerSec > 0 ? remaining / sweepsPerSec : 0;

            string line;
            if (showEnergy)
            {
                line = string.Format("\r{0} [{1}] {2,5:F1}% | {3}/{4} | E={5:F2} | Acc={6:F1}% | {7:F1} sw/s | ETA {8}   ",
                    phase, bar, pct * 100, current, total, energy, acceptRate * 100, sweepsPerSec, FormatElapsed(etaSec));
            }
            else
            {
                line = string.Format("\r{0} [{1}] {2,5:F1}% | {3}/{4} | {5:F1} sw/s | ETA {6}   ",
                    phase, bar, pct * 100, current, total, sweepsPerSec, FormatElapsed(etaSec));
            }

            Console.Write(line);
            Console.Out.Flush();
        }

        static string FormatElapsed(double totalSeconds)
        {
            if (totalSeconds < 0) totalSeconds = 0;
            int h = (int)(totalSeconds / 3600);
            int m = (int)((totalSeconds % 3600) / 60);
            int s = (int)(totalSeconds % 60);
            if (h > 0)
                return string.Format("{0}h{1:D2}m{2:D2}s", h, m, s);
            if (m > 0)
                return string.Format("{0}m{1:D2}s", m, s);
            return string.Format("{0}s", s);
        }

        static void ParseArguments(string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i].ToLowerInvariant();
                string next = (i + 1 < args.Length) ? args[i + 1] : null;

                switch (arg)
                {
                    case "--segment-mode":
                        if (next != null)
                        {
                            SegmentMode = next.ToLowerInvariant() == "fine"
                                ? SegmentModeEnum.Fine
                                : SegmentModeEnum.Coarse;
                            i++;
                        }
                        break;
                    case "--length":
                        if (next != null) { ChainLength = int.Parse(next); i++; }
                        break;
                    case "--chains":
                        if (next != null) { NumChains = int.Parse(next); i++; }
                        break;
                    case "--seed":
                        if (next != null) { Seed = int.Parse(next); i++; }
                        break;
                    case "--equilibration":
                        if (next != null) { EquilibrationSweeps = int.Parse(next); i++; }
                        break;
                    case "--production":
                        if (next != null) { ProductionSweeps = int.Parse(next); i++; }
                        break;
                    case "--output":
                        if (next != null) { OutputDir = next; i++; }
                        break;
                    case "--residues-per-segment":
                        if (next != null) { ResiduesPerSegment = int.Parse(next); i++; }
                        break;
                    case "--max-displacement":
                        if (next != null) { MaxDisplacement = double.Parse(next, CultureInfo.InvariantCulture); i++; }
                        break;
                    case "--move-size":
                        if (next != null) { MoveSize = int.Parse(next); i++; }
                        break;
                    case "--box-size":
                        if (next != null) { BoxSizeOverride = double.Parse(next, CultureInfo.InvariantCulture); i++; }
                        break;
                    case "--max-angle":
                        if (next != null) { MaxAngle = double.Parse(next, CultureInfo.InvariantCulture); i++; }
                        break;
                    case "--batch":
                        BatchMode = true;
                        break;
                }
            }
        }
    }
}
