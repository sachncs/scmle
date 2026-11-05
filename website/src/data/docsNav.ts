// Single source for the documentation sidebar, the pager and the overview page.
export interface NavPage { slug: string; label: string; summary: string; key: string }
export interface NavGroup { title: string; pages: NavPage[] }

export const docsNav: NavGroup[] = [
  {
    title: "Get started",
    pages: [
      { slug: "", label: "Overview", key: "start", summary: "Where to begin, and every page on this site grouped by what you are trying to do." },
      { slug: "quickstart", label: "Quickstart", key: "quickstart", summary: "Install the package and run your first curation-first project." },
      { slug: "data", label: "Data & curation", key: "data", summary: "Choose an input path, understand supported formats, and read the curation report." },
      { slug: "modalities", label: "Modality lifecycles", key: "modalities", summary: "Follow the 15-stage contract from raw input to an honest release boundary." },
      { slug: "capabilities", label: "Capability matrix", key: "capabilities", summary: "See which dataset families are supported, experimental, or extension-only." },
      { slug: "governed-data", label: "Governed data", key: "governed-data", summary: "Object stores with verified checksums, snapshot lineage, declared quality gates and incremental processing." },
    ],
  },
  {
    title: "Modelling and evidence",
    pages: [
      { slug: "modeling", label: "Modelling & estimation", key: "modeling", summary: "Understand the transformer paths, the JEPA boundary, uncertainty, and what parameter estimation still needs." },
      { slug: "evidence", label: "Evidence & gates", key: "evidence", summary: "Record what a claim means, what proves it, and what blocks a release." },
      { slug: "qualification", label: "Qualification instrument", key: "qualification", summary: "The signed charter, custody of held-out sets, the hypothesis ledger, tiers, and which detectors are certified." },
      { slug: "causal-decision", label: "Causal, decision & framing", key: "causal-decision", summary: "Frame a request, identify a causal effect or refuse, and choose a decision under stated costs." },
      { slug: "workflows", label: "Workflows", key: "workflows", summary: "Move from a project declaration to plans, runs, resumes, and predictions." },
      { slug: "cards", label: "Cards & explanations", key: "cards", summary: "Carry provenance, limitations, and review decisions with data and models." },
      { slug: "extensions", label: "Extension contracts", key: "extensions", summary: "Add readers, models, metrics and policies without editing the core." },
      { slug: "methods", label: "Methods checked", key: "methods", summary: "Which cited methods were read against their papers, and which routines match frozen scipy outputs." },
    ],
  },
  {
    title: "Reference",
    pages: [
      { slug: "api", label: "Python API", key: "api", summary: "Use the stable top-level objects and understand what is safe to depend on." },
      { slug: "api-stability", label: "API stability", key: "api-stability", summary: "What is pinned, which modules are extension surfaces, and how changes are recorded." },
      { slug: "cli", label: "Command line", key: "cli", summary: "Every command and the common run options." },
      { slug: "artifacts", label: "Artifacts & bundles", key: "artifacts", summary: "Every file a run, ingestion, framing pass, loop and service package writes." },
      { slug: "architecture", label: "Architecture", key: "architecture", summary: "See how ingestion, profiling, evaluation, packaging, and monitoring connect, and the known limits." },
      { slug: "dependencies", label: "Dependencies & extras", key: "dependencies", summary: "The light base install, every optional extra, version bounds and licence checks." },
      { slug: "benchmarks", label: "Benchmarks", key: "benchmarks", summary: "Full locked-test tables, repeated-seed intervals, new task types and the nine reliability checks." },
    ],
  },
  {
    title: "Serve and operate",
    pages: [
      { slug: "operations", label: "Operations", key: "operations", summary: "Resume interrupted work, verify bundles, approve promotions, and evolve safely." },
      { slug: "operating-loop", label: "Operating loop", key: "operating-loop", summary: "Measure, observe, evaluate, evolve: the evidence map and review cadence." },
      { slug: "service", label: "Serving contract & SLO", key: "service", summary: "Endpoints, schema, data policies, limits and declared objectives." },
      { slug: "local", label: "Local environment", key: "local", summary: "Run train, package, serve, parity and rollback on one machine." },
      { slug: "orchestration", label: "Orchestration", key: "orchestration", summary: "One workflow definition, a durable local executor, emitted cloud definitions." },
      { slug: "portable-training", label: "Portable training", key: "portable-training", summary: "One job spec across local, AWS and GCP runners, with checkpoints and cost budgets." },
      { slug: "recovery", label: "Recovery", key: "recovery", summary: "Keep failed runs observable and recover without promoting partial output." },
      { slug: "rehearsal", label: "Operational rehearsal", key: "rehearsal", summary: "Interruption, rollback, key rotation, fault evidence and monitoring handoff before a release." },
      { slug: "migrations", label: "Migrations", key: "migrations", summary: "Upgrade contracts deliberately and rerun the evidence that changed." },
      { slug: "troubleshooting", label: "Troubleshooting", key: "troubleshooting", summary: "Symptoms, causes and fixes." },
    ],
  },
  {
    title: "Cloud and release",
    pages: [
      { slug: "infrastructure", label: "Infrastructure as code", key: "infrastructure", summary: "The Terraform for AWS and GCP, the policy check, and destroy without orphans." },
      { slug: "cloud-release", label: "Cloud release", key: "cloud-release", summary: "Gate, live tests, signed record, promotion, rollback and reconcile." },
      { slug: "releasing", label: "Releasing the package", key: "releasing", summary: "Version, tag, preflight and the tag pipeline." },
    ],
  },
  {
    title: "Trust",
    pages: [
      { slug: "security", label: "Security", key: "security", summary: "Understand signatures, privacy, dependencies, and trust boundaries." },
      { slug: "threat-model", label: "Threat model & controls", key: "threat-model", summary: "The control behind each risk and the limits of custody." },
      { slug: "tenant-isolation", label: "Tenant isolation", key: "tenant-isolation", summary: "One deployment per tenant, scoped tokens, safe errors and audit." },
    ],
  },
];
