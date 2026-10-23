import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const API_BASE = process.env.EXPO_PUBLIC_API_BASE ?? "http://localhost:8000";
const LETTERS = ["A", "B", "C", "D"] as const;
const PRESSURE_TYPES = ["doubt", "authority", "wrong_suggestion", "correct_suggestion"] as const;

type Message = { role: string; content: string };

type Report = {
  protocol: string;
  baseline_accuracy: number | null;
  pressure_robust_accuracy: number | null;
  conditional_pressure_robustness: number | null;
  pressure_accuracy_mean: number | null;
  sycophancy_flip_to_wrong_rate: number | null;
  sycophancy_by_pressure_type: Record<string, number | null>;
  pressure_accuracy_by_type: Record<string, number | null>;
  answer_change_rate_by_type: Record<string, number | null>;
  wrong_suggestion_flip_rate: number | null;
  regressive_sycophancy_rate: number | null;
  pressure_other_error_rate: number | null;
  pressure_resistance_rate: number | null;
  corrective_update_rate: number | null;
  stubbornness_rate: number | null;
  correct_pressure_other_wrong_rate: number | null;
  correction_selectivity: number | null;
  strict_correction_selectivity: number | null;
  progressive_sycophancy_rate: number | null;
  regressive_sycophancy_any_wrong_rate: number | null;
  bidirectional_flip_rate: number | null;
  mean_user_belief_error_delta: number | null;
  answer_change_rate_wrong_pressure: number | null;
  harmful_flip_capture_ratio: number | null;
  parse_compliance_rate: number | null;
  exact_one_letter_compliance_rate: number | null;
  user_confidence_transfer_coefficient: number | null;
  domain_breakdown: Record<string, {
    trials: number;
    baseline_accuracy: number | null;
    sycophancy_macro_rate: number | null;
    wrong_suggestion_flip_rate: number | null;
    exact_wrong_suggestion_adoption_rate: number | null;
    pressure_robust_accuracy: number | null;
  }>;
  counts: Record<string, number>;
};

type EvidenceReport = {
  total_items: number;
  baseline_correct_items: number;
  baseline_accuracy: number | null;
  false_note_amplification_vs_claim_only: number | null;
  true_note_rescue_vs_claim_only: number | null;
  false_vs_true_note_with_wrong_claim: number | null;
  false_vs_true_note_without_claim: number | null;
  claim_evidence_interaction: number | null;
  flip_to_wrong_or_unparsed_rate: Record<string, number | null>;
  exact_wrong_suggestion_adoption_rate: Record<string, number | null>;
};

type EvidenceRun = {
  item_id: string;
  condition: string;
  messages: Message[];
  answer: string | null;
  correct_answer: string;
  raw_answer: string;
};

type EvidencePreview = { item_id: string; condition: string; messages: Message[] };

type RunRow = {
  run_id: string;
  item_id: string;
  variant_id: number;
  condition: string;
  pressure_type: string | null;
  messages: Message[];
  raw_answer?: string | null;
  answer: string | null;
  correct_answer: string;
  user_suggestion: string | null;
  classification: string;
  baseline_answer?: string | null;
};

type ManifestRow = {
  run_id: string;
  item_id: string;
  variant_id: number;
  condition: string;
  pressure_type: string | null;
  messages: Message[];
  correct_answer: string;
  user_suggestion: string | null;
  conditional?: boolean;
  conditional_note?: string;
};

type TestPromptItem = {
  item_id: string;
  domain: string;
  question: string;
  choices: Record<string, string>;
  correct_answer: string;
  wrong_suggestion: string;
};

type ProtocolConfig = {
  name: string;
  source_title: string;
  source_url: string;
  source_note: string;
  baseline_template: string;
  pressure_variants: Record<string, string[]>;
};

type TestPromptConfig = {
  version: number;
  protocol: ProtocolConfig;
  items: TestPromptItem[];
};

type MetricDefinition = {
  key: string;
  name: string;
  source: string;
  status: string;
  direction: string;
  definition: string;
};

type PerformanceRow = {
  case_id: string;
  category: string;
  repeat_index: number;
  grader_type: string;
  messages: Message[];
  raw_answer: string;
  passed: boolean;
  reason: string;
  confidence: number | null;
  elapsed_ms: number | null;
};

type PerformanceReport = {
  runs: number;
  unique_cases: number;
  overall_accuracy: number | null;
  structured_format_compliance: number | null;
  repeat_consistency: number | null;
  all_repeats_correct: number | null;
  repeated_cases: number;
  category_breakdown: Record<string, { runs: number; accuracy: number | null; cases: number }>;
  calibration: { samples: number; brier_score: number | null; ece_5_bins: number | null };
  latency_ms: { observed: number; mean: number | null; p50: number | null; p95: number | null };
  generation_tokens_per_second: { observed: number; mean: number | null };
};

type PerformancePreview = { case_id: string; category: string; repeat_index: number; grader_type: string; messages: Message[] };

type RagValue = { mean: number | null; samples: number };
type RagReport = {
  item_count: number;
  retrieval_at_k: Record<string, Record<string, RagValue>>;
  context_precision_ap: RagValue;
  generation: Record<string, RagValue>;
  rows: {
    item_id: string;
    question: string;
    judgment_source: string;
    retrieval: { gold_count: number | null; context_precision_ap: number | null };
    generation: Record<string, number | null>;
    judgment: unknown;
    judge_raw_response: string | null;
  }[];
};

function numberValue(value: number | null | undefined, unit = "", digits = 1) {
  return value == null ? "—" : `${value.toFixed(digits)}${unit}`;
}

function pct(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
function pp(value: number | null | undefined) {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)} pp`;
}

function Section({ children }: { children: React.ReactNode }) {
  return (
    <View style={{ backgroundColor: "white", borderRadius: 12, padding: 16, borderWidth: 1, borderColor: "#e1e5eb", gap: 12 }}>
      {children}
    </View>
  );
}

function Card({ title, value, note }: { title: string; value: string; note?: string }) {
  return (
    <View style={{ flexGrow: 1, flexBasis: 155, borderWidth: 1, borderColor: "#d6dbe3", borderRadius: 10, padding: 12, backgroundColor: "white" }}>
      <Text style={{ fontSize: 12, color: "#586174", marginBottom: 4 }}>{title}</Text>
      <Text style={{ fontSize: 24, fontWeight: "700", color: "#111827" }}>{value}</Text>
      {note ? <Text style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>{note}</Text> : null}
    </View>
  );
}

function LabeledInput({ label, value, onChangeText, multiline = false }: { label: string; value: string; onChangeText: (value: string) => void; multiline?: boolean }) {
  return (
    <View style={{ gap: 4 }}>
      <Text style={{ fontWeight: "600" }}>{label}</Text>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        multiline={multiline}
        autoCapitalize="none"
        style={{ borderWidth: 1, borderColor: "#cbd2dc", borderRadius: 8, paddingHorizontal: 10, paddingVertical: 9, minHeight: multiline ? 82 : undefined, textAlignVertical: multiline ? "top" : "center" }}
      />
    </View>
  );
}

function Conversation({ messages, rawAnswer }: { messages: Message[]; rawAnswer?: string | null }) {
  return (
    <View style={{ borderWidth: 1, borderColor: "#dbe2ea", borderRadius: 8, overflow: "hidden" }}>
      {messages.map((message, index) => (
        <View key={`${message.role}-${index}`} style={{ borderTopWidth: index ? 1 : 0, borderTopColor: "#dbe2ea" }}>
          <Text style={{ paddingHorizontal: 10, paddingVertical: 6, fontSize: 11, fontWeight: "700", color: message.role === "assistant" ? "#7c3aed" : "#1d4ed8", backgroundColor: "#f8fafc" }}>
            {message.role.toUpperCase()}
          </Text>
          <Text selectable style={{ padding: 10, fontFamily: "monospace", fontSize: 13, lineHeight: 19 }}>{message.content}</Text>
        </View>
      ))}
      {rawAnswer != null ? (
        <View style={{ borderTopWidth: 1, borderTopColor: "#dbe2ea" }}>
          <Text style={{ paddingHorizontal: 10, paddingVertical: 6, fontSize: 11, fontWeight: "700", color: "#166534", backgroundColor: "#f0fdf4" }}>MODEL RESPONSE</Text>
          <Text selectable style={{ padding: 10, fontFamily: "monospace", fontSize: 13 }}>{rawAnswer}</Text>
        </View>
      ) : null}
    </View>
  );
}

export default function HomeScreen() {
  const [model, setModel] = useState("llama3.1:8b");
  const [evidenceCount, setEvidenceCount] = useState("4");
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [evidenceReport, setEvidenceReport] = useState<EvidenceReport | null>(null);
  const [evidenceRuns, setEvidenceRuns] = useState<EvidenceRun[]>([]);
  const [evidencePreview, setEvidencePreview] = useState<EvidencePreview[]>([]);
  const [showEvidenceRuns, setShowEvidenceRuns] = useState(false);

  const [itemCount, setItemCount] = useState("4");
  const [variantCount, setVariantCount] = useState("3");
  const [includeCorrectControl, setIncludeCorrectControl] = useState(true);
  const [report, setReport] = useState<Report | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [previewRows, setPreviewRows] = useState<ManifestRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [promptConfig, setPromptConfig] = useState<TestPromptConfig | null>(null);
  const [promptLoading, setPromptLoading] = useState(true);
  const [promptSaving, setPromptSaving] = useState(false);
  const [promptStatus, setPromptStatus] = useState<string | null>(null);
  const [metricDefinitions, setMetricDefinitions] = useState<MetricDefinition[]>([]);
  const [performanceCategories, setPerformanceCategories] = useState<string[]>([]);
  const [performanceCaseCounts, setPerformanceCaseCounts] = useState<Record<string, number>>({});
  const [performanceCategory, setPerformanceCategory] = useState("all");
  const [performanceCount, setPerformanceCount] = useState("12");
  const [performanceRepetitions, setPerformanceRepetitions] = useState("1");
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [performanceReport, setPerformanceReport] = useState<PerformanceReport | null>(null);
  const [performanceRows, setPerformanceRows] = useState<PerformanceRow[]>([]);
  const [performancePreview, setPerformancePreview] = useState<PerformancePreview[]>([]);
  const [performanceShowAll, setPerformanceShowAll] = useState(false);
  const [ragJson, setRagJson] = useState("");
  const [ragJudgeModel, setRagJudgeModel] = useState("llama3.1:8b");
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState<string | null>(null);
  const [ragReport, setRagReport] = useState<RagReport | null>(null);
  const [ragShowEvidence, setRagShowEvidence] = useState(false);

  const counts = useMemo(() => {
    const n = Number.parseInt(itemCount || "0", 10) || 0;
    const v = Number.parseInt(variantCount || "0", 10) || 0;
    const minimum = n * v * 4; // baseline + 3 misleading pressure calls
    const maximum = minimum + (includeCorrectControl ? n * v : 0);
    return { minimum, maximum };
  }, [itemCount, variantCount, includeCorrectControl]);

  const experimentRequest = () => {
    const item_count = Number.parseInt(evidenceCount, 10);
    if (!Number.isInteger(item_count) || item_count < 1 || item_count > 8) {
      throw new Error("Evidence demo needs 1–8 items.");
    }
    return { item_count, model };
  };

  const callEvidence = async (previewOnly: boolean) => {
    setEvidenceLoading(true);
    setEvidenceError(null);
    setEvidenceReport(null);
    setEvidenceRuns([]);
    setEvidencePreview([]);
    try {
      const response = await fetch(`${API_BASE}/evidence/${previewOnly ? "preview" : "run"}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(experimentRequest()),
      });
      const content = await response.text();
      if (!response.ok) throw new Error(`${response.status}: ${content}`);
      const data = JSON.parse(content);
      if (previewOnly) setEvidencePreview(data.manifest ?? []);
      else { setEvidenceReport(data.report); setEvidenceRuns(data.runs ?? []); }
    } catch (err: any) {
      setEvidenceError(err?.message ?? String(err));
    } finally {
      setEvidenceLoading(false);
    }
  };

  const loadConfig = async () => {
    setPromptLoading(true);
    try {
      const [promptRes, metricRes] = await Promise.all([
        fetch(`${API_BASE}/benchmark/test-prompts`),
        fetch(`${API_BASE}/benchmark/metric-definitions`),
      ]);
      const promptBody = await promptRes.text();
      const metricBody = await metricRes.text();
      if (!promptRes.ok) throw new Error(`${promptRes.status}: ${promptBody}`);
      if (!metricRes.ok) throw new Error(`${metricRes.status}: ${metricBody}`);
      setPromptConfig(JSON.parse(promptBody));
      setMetricDefinitions(JSON.parse(metricBody).metrics ?? []);
    } catch (err: any) {
      setPromptStatus(err?.message ?? String(err));
    } finally {
      setPromptLoading(false);
    }
  };

  useEffect(() => { loadConfig(); }, []);

  useEffect(() => {
    fetch(`${API_BASE}/performance/cases`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Could not fetch performance cases (${response.status})`);
        return response.json();
      })
      .then((data) => {
        setPerformanceCategories(data.categories ?? []);
        const counts: Record<string, number> = { all: data.count ?? 0 };
        for (const entry of data.cases ?? []) counts[entry.category] = (counts[entry.category] ?? 0) + 1;
        setPerformanceCaseCounts(counts);
      })
      .catch((err) => setPerformanceError(String(err)));
  }, []);

  const loadRagExamples = async () => {
    setRagError(null);
    try {
      const res = await fetch(`${API_BASE}/rag/examples`);
      if (!res.ok) throw new Error(`Examples unavailable (${res.status})`);
      setRagJson(JSON.stringify(await res.json(), null, 2));
      setRagReport(null);
    } catch (err: any) { setRagError(err?.message ?? String(err)); }
  };
  useEffect(() => { loadRagExamples(); }, []);

  const runRag = async (useJudge: boolean) => {
    setRagError(null); setRagReport(null); setRagLoading(true);
    try {
      const request = JSON.parse(ragJson);
      if (useJudge) request.judge_model = ragJudgeModel;
      const result = await fetch(`${API_BASE}/rag/${useJudge ? "judge" : "score"}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
      });
      const body = await result.text();
      if (!result.ok) throw new Error(`${result.status}: ${body}`);
      setRagReport(JSON.parse(body));
    } catch (err: any) { setRagError(err?.message ?? String(err)); }
    finally { setRagLoading(false); }
  };

  const callPerformance = async (previewOnly: boolean) => {
    const item_count = Number(performanceCount);
    const repetitions = Number(performanceRepetitions);
    if (!Number.isInteger(item_count) || item_count < 1 || item_count > 100 || !Number.isInteger(repetitions) || repetitions < 1 || repetitions > 5) {
      setPerformanceError("Enter 1–100 cases and 1–5 repetitions.");
      return;
    }
    setPerformanceLoading(true);
    setPerformanceError(null);
    setPerformanceReport(null);
    setPerformanceRows([]);
    setPerformancePreview([]);
    try {
      const categories = performanceCategory === "all" ? null : [performanceCategory];
      const availableCount = performanceCaseCounts[performanceCategory] ?? item_count;
      const selectedCount = Math.min(item_count, availableCount);
      if (selectedCount < 1) throw new Error("No cases available for this category.");
      const response = await fetch(`${API_BASE}/performance/${previewOnly ? "preview" : "run"}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_count: selectedCount, repetitions, categories, model }),
      });
      const body = await response.text();
      if (!response.ok) throw new Error(`${response.status}: ${body}`);
      const data = JSON.parse(body);
      if (previewOnly) setPerformancePreview(data.manifest ?? []);
      else { setPerformanceReport(data.report); setPerformanceRows(data.runs ?? []); }
    } catch (err: any) {
      setPerformanceError(err?.message ?? String(err));
    } finally {
      setPerformanceLoading(false);
    }
  };

  const saveTestPrompts = async () => {
    if (!promptConfig) return;
    setPromptSaving(true);
    setPromptStatus(null);
    try {
      const response = await fetch(`${API_BASE}/benchmark/test-prompts`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(promptConfig),
      });
      const body = await response.text();
      if (!response.ok) throw new Error(`${response.status}: ${body}`);
      const data = JSON.parse(body);
      setPromptConfig(data.config);
      setPromptStatus(`Saved ${data.count} test items.`);
    } catch (err: any) {
      setPromptStatus(err?.message ?? String(err));
    } finally {
      setPromptSaving(false);
    }
  };

  const updateProtocol = (patch: Partial<ProtocolConfig>) => setPromptConfig((current) => current ? ({ ...current, protocol: { ...current.protocol, ...patch } }) : current);
  const updateVariant = (type: string, index: number, value: string) => setPromptConfig((current) => {
    if (!current) return current;
    const pressure = { ...current.protocol.pressure_variants };
    const variants = [...(pressure[type] ?? [])];
    variants[index] = value;
    pressure[type] = variants;
    return { ...current, protocol: { ...current.protocol, pressure_variants: pressure } };
  });
  const updatePromptItem = (index: number, patch: Partial<TestPromptItem>) => setPromptConfig((current) => {
    if (!current) return current;
    const items = [...current.items];
    items[index] = { ...items[index], ...patch };
    return { ...current, items };
  });
  const updateChoice = (index: number, letter: string, value: string) => setPromptConfig((current) => {
    if (!current) return current;
    const items = [...current.items];
    items[index] = { ...items[index], choices: { ...items[index].choices, [letter]: value } };
    return { ...current, items };
  });
  const addPromptItem = () => setPromptConfig((current) => {
    if (!current) return current;
    let serial = current.items.length + 1;
    const ids = new Set(current.items.map((item) => item.item_id));
    let itemId = `custom_${String(serial).padStart(3, "0")}`;
    while (ids.has(itemId)) { serial += 1; itemId = `custom_${String(serial).padStart(3, "0")}`; }
    return { ...current, items: [...current.items, { item_id: itemId, domain: "custom", question: "", choices: { A: "", B: "", C: "", D: "" }, correct_answer: "A", wrong_suggestion: "B" }] };
  });
  const removePromptItem = (index: number) => setPromptConfig((current) => current && current.items.length > 1 ? ({ ...current, items: current.items.filter((_, i) => i !== index) }) : current);

  const requestShape = () => {
    const item_count = Number.parseInt(itemCount, 10);
    const variant_count = Number.parseInt(variantCount, 10);
    if (!Number.isFinite(item_count) || item_count < 1) throw new Error("Item count must be at least 1.");
    if (!Number.isFinite(variant_count) || variant_count < 1 || variant_count > 3) throw new Error("Variant count must be 1, 2, or 3.");
    return { item_count, variant_count, include_correct_control: includeCorrectControl };
  };

  const previewPrompts = async () => {
    setPreviewLoading(true); setError(null); setPreviewRows([]);
    try {
      const response = await fetch(`${API_BASE}/benchmark/preview`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestShape()) });
      const body = await response.text();
      if (!response.ok) throw new Error(`${response.status}: ${body}`);
      setPreviewRows(JSON.parse(body).manifest ?? []);
    } catch (err: any) { setError(err?.message ?? String(err)); }
    finally { setPreviewLoading(false); }
  };

  const runBenchmark = async () => {
    setLoading(true); setError(null); setReport(null); setRuns([]);
    try {
      const response = await fetch(`${API_BASE}/benchmark/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...requestShape(), model }) });
      const body = await response.text();
      if (!response.ok) throw new Error(`${response.status}: ${body}`);
      const data = JSON.parse(body);
      setReport(data.report);
      setRuns(data.runs ?? []);
    } catch (err: any) { setError(err?.message ?? String(err)); }
    finally { setLoading(false); }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#f4f6f8" }}>
      <ScrollView contentContainerStyle={{ padding: 16, alignItems: "center" }}>
        <View style={{ width: "100%", maxWidth: 1050, gap: 14 }}>
          <View>
            <Text style={{ fontSize: 30, fontWeight: "800" }}>sycophancy-evaluation</Text>
            <Text style={{ color: "#5f6878", marginTop: 4 }}>Auditable sycophancy experiments and broader LLM performance diagnostics.</Text>
          </View>

          {promptConfig ? (
            <Section>
              <Text style={{ fontSize: 18, fontWeight: "700" }}>Default protocol</Text>
              <Text style={{ fontWeight: "700" }}>{promptConfig.protocol.name}</Text>
              <Text style={{ color: "#4b5563" }}>{promptConfig.protocol.source_title}</Text>
              <Text selectable style={{ color: "#1d4ed8", fontSize: 12 }}>{promptConfig.protocol.source_url}</Text>
              <Text style={{ color: "#4b5563" }}>{promptConfig.protocol.source_note}</Text>
              <View style={{ backgroundColor: "#ecfdf5", borderRadius: 8, padding: 10 }}>
                <Text style={{ color: "#166534", fontWeight: "700" }}>No evaluation-oriented system prompt is injected.</Text>
                <Text style={{ color: "#166534", marginTop: 3 }}>Pressure is a follow-up user turn after the model has already answered the baseline question.</Text>
              </View>
            </Section>
          ) : null}

          <Section>
            <Text style={{ fontSize: 18, fontWeight: "700" }}>General LLM performance diagnostics</Text>
            <Text style={{ color: "#4b5563" }}>Independent of the sycophancy protocol. Original, deterministic tasks cover arithmetic, reasoning, extraction, JSON, instruction following, grounding, multilingual vocabulary and more. These are small diagnostic checks, not a standardized leaderboard.</Text>
            <Text style={{ fontWeight: "700" }}>Capability category</Text>
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
              {["all", ...performanceCategories].map((category) => (
                <Pressable key={category} onPress={() => setPerformanceCategory(category)} style={{ borderRadius: 7, paddingHorizontal: 9, paddingVertical: 6, borderWidth: 1, borderColor: performanceCategory === category ? "#1d4ed8" : "#cbd5e1", backgroundColor: performanceCategory === category ? "#dbeafe" : "white" }}>
                  <Text style={{ fontSize: 12, color: "#1e3a8a" }}>{category.replaceAll("_", " ")}</Text>
                </Pressable>
              ))}
            </View>
            <LabeledInput label={`Number of cases (available: ${performanceCaseCounts[performanceCategory] ?? "loading…"})`} value={performanceCount} onChangeText={setPerformanceCount} />
            <LabeledInput label="Repetitions per case (1–5)" value={performanceRepetitions} onChangeText={setPerformanceRepetitions} />
            <LabeledInput label="Ollama model (shared with SycoBench)" value={model} onChangeText={setModel} />
            <Text style={{ color: "#6b7280", fontSize: 12 }}>Small runs spread cases across categories. Repetitions assess exact response consistency; temperature defaults to zero. Higher case count and repetition count increase processing time.</Text>
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
              <Pressable onPress={() => callPerformance(true)} disabled={performanceLoading} style={{ flexGrow: 1, padding: 11, borderRadius: 8, borderWidth: 1, borderColor: "#1d4ed8", alignItems: "center" }}>
                <Text style={{ color: "#1d4ed8", fontWeight: "700" }}>Preview tasks</Text>
              </Pressable>
              <Pressable onPress={() => callPerformance(false)} disabled={performanceLoading} style={{ flexGrow: 1, padding: 11, borderRadius: 8, backgroundColor: "#1d4ed8", alignItems: "center" }}>
                <Text style={{ color: "white", fontWeight: "700" }}>{performanceLoading ? "Evaluating…" : "Run performance suite"}</Text>
              </Pressable>
            </View>
            {performanceLoading ? <ActivityIndicator /> : null}
            {performanceError ? <Text style={{ color: "#b91c1c" }}>{performanceError}</Text> : null}
            {performancePreview.length ? (
              <View style={{ gap: 8 }}>
                <Text style={{ fontWeight: "700" }}>Task preview ({performancePreview.length} model calls)</Text>
                {performancePreview.map((row) => (
                  <View key={`${row.case_id}-${row.repeat_index}`} style={{ gap: 4 }}>
                    <Text style={{ fontWeight: "600" }}>{row.category} / {row.case_id} / repeat {row.repeat_index + 1} / {row.grader_type}</Text>
                    <Conversation messages={row.messages} />
                  </View>
                ))}
              </View>
            ) : null}
            {performanceReport ? (
              <View style={{ gap: 10 }}>
                <Text>{performanceReport.unique_cases} cases · {performanceReport.runs} calls · deterministic local rubric</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                  <Card title="Accuracy" value={pct(performanceReport.overall_accuracy)} />
                  <Card title="Structured format" value={pct(performanceReport.structured_format_compliance)} />
                  <Card title="Exact repeat consistency" value={pct(performanceReport.repeat_consistency)} note={`Across ${performanceReport.repeated_cases} repeated cases`} />
                  <Card title="All repetitions correct" value={pct(performanceReport.all_repeats_correct)} />
                  <Card title="Median latency" value={numberValue(performanceReport.latency_ms.p50, " ms", 0)} />
                  <Card title="Mean generation speed" value={numberValue(performanceReport.generation_tokens_per_second.mean, " tok/s")} />
                  <Card title="Brier score ↓" value={numberValue(performanceReport.calibration.brier_score, "", 3)} note={`${performanceReport.calibration.samples} parseable confidence responses`} />
                  <Card title="ECE (5 bins) ↓" value={numberValue(performanceReport.calibration.ece_5_bins, "", 3)} note="Exploratory at small N" />
                </View>
                <Text style={{ fontWeight: "700" }}>Category accuracy (each denominator shown)</Text>
                {Object.entries(performanceReport.category_breakdown).map(([category, values]) => (
                  <View key={category} style={{ flexDirection: "row", justifyContent: "space-between", gap: 8 }}>
                    <Text style={{ flexShrink: 1 }}>{category.replaceAll("_", " ")} ({values.runs} calls)</Text>
                    <Text>{pct(values.accuracy)}</Text>
                  </View>
                ))}
                <Text style={{ color: "#64748b", fontSize: 12 }}>Exact-match and keyword checks are intentionally narrow. Response quality, safety, multimodal ability and complex software engineering require separate validated datasets and/or human judges. Missing runtime metadata is displayed as —.</Text>
                <Pressable onPress={() => setPerformanceShowAll(v => !v)} style={{ padding: 10, borderRadius: 7, borderWidth: 1, borderColor: "#cbd5e1" }}>
                  <Text style={{ color: "#1d4ed8" }}>{performanceShowAll ? "Hide" : "Show"} raw outputs and rubrics ({performanceRows.length})</Text>
                </Pressable>
                {performanceShowAll ? performanceRows.map((row) => (
                  <View key={`${row.case_id}-${row.repeat_index}`} style={{ gap: 4 }}>
                    <Text style={{ fontWeight: "700" }}>{row.case_id} · repeat {row.repeat_index + 1} · {row.passed ? "PASS" : "FAIL"} · {row.reason}</Text>
                    <Conversation messages={row.messages} rawAnswer={row.raw_answer} />
                  </View>
                )) : null}
              </View>
            ) : null}
          </Section>

          <Section>
            <Text style={{ fontSize: 18, fontWeight: "700" }}>RAG evaluation</Text>
            <Text style={{ color: "#4b5563" }}>Score retrieval traces from any RAG system. Retrieval metrics use reference document IDs; generation metrics use explicitly supplied claim annotations or an Ollama judge. No search index or external API is required.</Text>
            <Text style={{ color: "#92400e", fontSize: 12 }}>LLM-judged scores are estimates, not ground truth. For reliable comparisons, use an independent judge model and review supporting evidence.</Text>
            <Pressable onPress={loadRagExamples} disabled={ragLoading} style={{ alignSelf: "flex-start", padding: 8, borderWidth: 1, borderColor: "#94a3b8", borderRadius: 7 }}>
              <Text style={{ color: "#1d4ed8", fontWeight: "700" }}>Load two annotated examples</Text>
            </Pressable>
            <Text style={{ fontWeight: "700" }}>RAG traces JSON</Text>
            <TextInput
              multiline value={ragJson} onChangeText={setRagJson} autoCapitalize="none" autoCorrect={false}
              style={{ borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 8, padding: 12, minHeight: 180, fontFamily: "monospace", fontSize: 12, textAlignVertical: "top" }}
            />
            <LabeledInput label="Independent Ollama judge model" value={ragJudgeModel} onChangeText={setRagJudgeModel} />
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
              <Pressable onPress={() => runRag(false)} disabled={ragLoading || !ragJson.trim()} style={{ flexGrow: 1, padding: 11, borderRadius: 8, borderWidth: 1, borderColor: "#1d4ed8", alignItems: "center" }}>
                <Text style={{ color: "#1d4ed8", fontWeight: "700" }}>Score document IDs + annotations</Text>
              </Pressable>
              <Pressable onPress={() => runRag(true)} disabled={ragLoading || !ragJson.trim()} style={{ flexGrow: 1, padding: 11, borderRadius: 8, backgroundColor: "#1d4ed8", alignItems: "center" }}>
                <Text style={{ color: "white", fontWeight: "700" }}>{ragLoading ? "Evaluating…" : "Judge claims with Ollama"}</Text>
              </Pressable>
            </View>
            {ragLoading ? <ActivityIndicator /> : null}
            {ragError ? <Text style={{ color: "#b91c1c" }}>{ragError}</Text> : null}
            {ragReport ? (
              <View style={{ gap: 10 }}>
                <Text style={{ fontWeight: "700" }}>{ragReport.item_count} RAG evaluation items</Text>
                <Text style={{ fontSize: 15, fontWeight: "700" }}>Retrieval by K</Text>
                {Object.entries(ragReport.retrieval_at_k).map(([k, metrics]) => (
                  <View key={k} style={{ borderWidth: 1, borderColor: "#e2e8f0", borderRadius: 8, padding: 10, gap: 3 }}>
                    <Text style={{ fontWeight: "700" }}>K={k} (grounded queries: {metrics.recall.samples})</Text>
                    <Text>Recall {pct(metrics.recall.mean)} · Precision {pct(metrics.precision.mean)} · Hit {pct(metrics.hit_rate.mean)}</Text>
                    <Text>MRR {numberValue(metrics.mrr.mean, "", 3)} · nDCG {numberValue(metrics.ndcg.mean, "", 3)} · MAP {numberValue(metrics.average_precision.mean, "", 3)}</Text>
                  </View>
                ))}
                <Card title="Context precision (ranked AP)" value={numberValue(ragReport.context_precision_ap.mean, "", 3)} note={`N=${ragReport.context_precision_ap.samples}; based on provided relevance IDs`} />
                <Text style={{ fontSize: 15, fontWeight: "700" }}>Generation and context grounding</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                  {Object.entries(ragReport.generation).map(([key, value]) => (
                    <Card key={key} title={key.replaceAll("_", " ")} value={pct(value.mean)} note={`N=${value.samples}${key.startsWith("noise_") || key === "paired_noise_flip" ? " · lower is better" : ""}`} />
                  ))}
                </View>
                <Text style={{ color: "#64748b", fontSize: 12 }}>— means no valid reference, matching gold IDs or annotated/judged claims. These metrics are never inferred from document text by the deterministic scorer.</Text>
                <Pressable onPress={() => setRagShowEvidence(v => !v)} style={{ padding: 10, borderRadius: 7, borderWidth: 1, borderColor: "#cbd5e1" }}>
                  <Text style={{ color: "#1d4ed8" }}>{ragShowEvidence ? "Hide" : "Show"} judgment provenance and claims</Text>
                </Pressable>
                {ragShowEvidence ? ragReport.rows.map((row) => (
                  <View key={row.item_id} style={{ gap: 5, borderBottomWidth: 1, borderBottomColor: "#e2e8f0", paddingBottom: 10 }}>
                    <Text style={{ fontWeight: "700" }}>{row.item_id} · {row.judgment_source}</Text>
                    <Text>{row.question}</Text>
                    <Text>Faithfulness {pct(row.generation.faithfulness)} · Contextual recall {pct(row.generation.contextual_recall)} · Context precision {numberValue(row.retrieval.context_precision_ap, "", 3)}</Text>
                    <Text selectable style={{ fontFamily: "monospace", fontSize: 11 }}>{JSON.stringify(row.judgment, null, 2)}</Text>
                    {row.judge_raw_response ? <Text selectable style={{ fontSize: 11, color: "#64748b" }}>Raw judge response: {row.judge_raw_response}</Text> : null}
                  </View>
                )) : null}
              </View>
            ) : null}
          </Section>

          <Section>
            <Text style={{ fontSize: 18, fontWeight: "700" }}>Experimental evidence counterfactuals</Text>
            <Text style={{ color: "#4b5563" }}>
              New: replay the same correct baseline under a wrong user claim and factually true or false
              user-supplied notes. This is an exploratory paired experiment, not a SycoBench-600 score.
            </Text>
            <LabeledInput label="Evidence demo items (1–8)" value={evidenceCount} onChangeText={setEvidenceCount} />
            <Text style={{ color: "#6b7280", fontSize: 12 }}>
              Six model calls per item (one baseline plus five independent conditions).
              Notes are synthetic examples, not verified third-party citations.
            </Text>
            <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap" }}>
              <Pressable onPress={() => callEvidence(true)} disabled={evidenceLoading} style={{ flexGrow: 1, padding: 11, borderRadius: 8, borderWidth: 1, borderColor: "#1d4ed8", alignItems: "center" }}>
                <Text style={{ color: "#1d4ed8", fontWeight: "700" }}>Preview conditions</Text>
              </Pressable>
              <Pressable onPress={() => callEvidence(false)} disabled={evidenceLoading} style={{ flexGrow: 1, padding: 11, borderRadius: 8, backgroundColor: "#1d4ed8", alignItems: "center" }}>
                <Text style={{ color: "white", fontWeight: "700" }}>{evidenceLoading ? "Working…" : "Run evidence experiment"}</Text>
              </Pressable>
            </View>
            {evidenceError ? <Text style={{ color: "#b91c1c" }}>{evidenceError}</Text> : null}
            {evidencePreview.length > 0 ? (
              <View style={{ gap: 8 }}>
                <Text style={{ fontWeight: "700" }}>Preview ({evidencePreview.length} conversations; first case shown)</Text>
                {evidencePreview.slice(0, 6).map((row) => (
                  <View key={`${row.item_id}-${row.condition}`} style={{ gap: 4 }}>
                    <Text style={{ fontWeight: "600" }}>{row.condition}</Text>
                    <Conversation messages={row.messages} />
                  </View>
                ))}
              </View>
            ) : null}
            {evidenceReport ? (
              <View style={{ gap: 10 }}>
                <Text>Baseline correct: {evidenceReport.baseline_correct_items} / {evidenceReport.total_items}. Contrasts use only baseline-correct cases.</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                  <Card title="False-note amplification" value={pp(evidenceReport.false_note_amplification_vs_claim_only)} />
                  <Card title="True-note rescue" value={pp(evidenceReport.true_note_rescue_vs_claim_only)} />
                  <Card title="Claim × evidence interaction" value={pp(evidenceReport.claim_evidence_interaction)} />
                </View>
                <Text style={{ fontWeight: "700" }}>Flip to wrong or unparsed, by condition</Text>
                {Object.entries(evidenceReport.flip_to_wrong_or_unparsed_rate).map(([type, value]) => (
                  <View key={type} style={{ flexDirection: "row", justifyContent: "space-between", gap: 10 }}>
                    <Text style={{ flexShrink: 1 }}>{type}</Text><Text>{pct(value)}</Text>
                  </View>
                ))}
                <Pressable onPress={() => setShowEvidenceRuns(v => !v)} style={{ padding: 10, borderRadius: 7, borderWidth: 1, borderColor: "#cbd5e1" }}>
                  <Text style={{ color: "#1d4ed8" }}>{showEvidenceRuns ? "Hide" : "Show"} all raw evidence conversations ({evidenceRuns.length})</Text>
                </Pressable>
                {showEvidenceRuns ? evidenceRuns.map((run) => (
                  <View key={`${run.item_id}-${run.condition}`} style={{ gap: 4 }}>
                    <Text style={{ fontWeight: "600" }}>{run.item_id} · {run.condition} · answer={run.answer ?? "unparsed"} · truth={run.correct_answer}</Text>
                    <Conversation messages={run.messages} rawAnswer={run.raw_answer} />
                  </View>
                )) : null}
              </View>
            ) : null}
          </Section>

          <Section>
            <Text style={{ fontSize: 18, fontWeight: "700" }}>Evaluation definitions</Text>
            {metricDefinitions.map((metric) => (
              <View key={metric.key} style={{ borderBottomWidth: 1, borderBottomColor: "#edf0f4", paddingVertical: 8 }}>
                <Text style={{ fontWeight: "700" }}>{metric.name}</Text>
                <Text style={{ color: "#64748b", fontSize: 11 }}>{metric.source} · {metric.status}</Text>
                <Text style={{ color: "#4b5563", marginTop: 3 }}>{metric.definition}</Text>
              </View>
            ))}
          </Section>

          <Section>
            <View style={{ flexDirection: "row", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
              <View>
                <Text style={{ fontSize: 18, fontWeight: "700" }}>Test prompt definitions</Text>
                <Text style={{ color: "#6b7280", fontSize: 12, marginTop: 3 }}>Stored in backend/data/test_prompts.json. The paper templates and test items are editable.</Text>
              </View>
              <View style={{ flexDirection: "row", gap: 8 }}>
                <Pressable onPress={addPromptItem} disabled={!promptConfig || promptSaving} style={{ borderWidth: 1, borderColor: "#1d4ed8", borderRadius: 7, paddingHorizontal: 10, paddingVertical: 7 }}><Text style={{ color: "#1d4ed8", fontWeight: "700" }}>Add item</Text></Pressable>
                <Pressable onPress={saveTestPrompts} disabled={!promptConfig || promptSaving} style={{ backgroundColor: "#1d4ed8", borderRadius: 7, paddingHorizontal: 10, paddingVertical: 7 }}><Text style={{ color: "white", fontWeight: "700" }}>{promptSaving ? "Saving..." : "Save prompts"}</Text></Pressable>
              </View>
            </View>
            {promptLoading ? <ActivityIndicator /> : null}
            {promptStatus ? <Text style={{ color: promptStatus.startsWith("Saved") ? "#166534" : "#b91c1c" }}>{promptStatus}</Text> : null}
            {promptConfig ? (
              <>
                <LabeledInput label="Baseline prompt template" value={promptConfig.protocol.baseline_template} multiline onChangeText={(baseline_template) => updateProtocol({ baseline_template })} />
                <Text style={{ color: "#6b7280", fontSize: 12 }}>Placeholders: {"{question}"}, {"{A}"}, {"{B}"}, {"{C}"}, {"{D}"}.</Text>
                {PRESSURE_TYPES.map((type) => (
                  <View key={type} style={{ gap: 7 }}>
                    <Text style={{ fontSize: 15, fontWeight: "700" }}>{type.replaceAll("_", " ")}</Text>
                    {(promptConfig.protocol.pressure_variants[type] ?? []).map((value, index) => (
                      <LabeledInput key={`${type}-${index}`} label={`Variant ${index + 1}`} value={value} multiline onChangeText={(text) => updateVariant(type, index, text)} />
                    ))}
                  </View>
                ))}
                <Text style={{ color: "#6b7280", fontSize: 12 }}>wrong_suggestion requires {"{SUGGEST}"}; correct_suggestion requires {"{CORRECT}"}.</Text>

                <Text style={{ fontSize: 15, fontWeight: "700", marginTop: 4 }}>Test items ({promptConfig.items.length})</Text>
                {promptConfig.items.map((item, index) => (
                  <View key={`${item.item_id}-${index}`} style={{ borderWidth: 1, borderColor: "#dbe2ea", borderRadius: 9, padding: 12, gap: 8 }}>
                    <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                      <Text style={{ fontWeight: "800" }}>#{index + 1} {item.item_id}</Text>
                      <Pressable onPress={() => removePromptItem(index)} disabled={promptConfig.items.length <= 1} style={{ borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 6, paddingHorizontal: 8, paddingVertical: 5 }}><Text style={{ fontSize: 12 }}>Remove</Text></Pressable>
                    </View>
                    <LabeledInput label="Item ID" value={item.item_id} onChangeText={(item_id) => updatePromptItem(index, { item_id })} />
                    <LabeledInput label="Domain" value={item.domain} onChangeText={(domain) => updatePromptItem(index, { domain })} />
                    <LabeledInput label="Question" value={item.question} multiline onChangeText={(question) => updatePromptItem(index, { question })} />
                    {LETTERS.map((letter) => <LabeledInput key={letter} label={`Choice ${letter}`} value={item.choices[letter] ?? ""} onChangeText={(value) => updateChoice(index, letter, value)} />)}
                    <View style={{ flexDirection: "row", gap: 14, flexWrap: "wrap" }}>
                      <View style={{ gap: 5 }}><Text style={{ fontWeight: "600" }}>Correct answer</Text><View style={{ flexDirection: "row", gap: 6 }}>{LETTERS.map((letter) => <Pressable key={letter} onPress={() => updatePromptItem(index, { correct_answer: letter })} style={{ borderWidth: 1, borderColor: item.correct_answer === letter ? "#1d4ed8" : "#cbd5e1", backgroundColor: item.correct_answer === letter ? "#dbeafe" : "white", borderRadius: 6, paddingHorizontal: 10, paddingVertical: 6 }}><Text style={{ fontWeight: "700" }}>{letter}</Text></Pressable>)}</View></View>
                      <View style={{ gap: 5 }}><Text style={{ fontWeight: "600" }}>Known-wrong suggestion</Text><View style={{ flexDirection: "row", gap: 6 }}>{LETTERS.map((letter) => <Pressable key={letter} onPress={() => updatePromptItem(index, { wrong_suggestion: letter })} style={{ borderWidth: 1, borderColor: item.wrong_suggestion === letter ? "#b91c1c" : "#cbd5e1", backgroundColor: item.wrong_suggestion === letter ? "#fee2e2" : "white", borderRadius: 6, paddingHorizontal: 10, paddingVertical: 6 }}><Text style={{ fontWeight: "700" }}>{letter}</Text></Pressable>)}</View></View>
                    </View>
                  </View>
                ))}
              </>
            ) : null}
          </Section>

          <Section>
            <Text style={{ fontSize: 18, fontWeight: "700" }}>Experiment</Text>
            <LabeledInput label="Ollama model" value={model} onChangeText={setModel} />
            <LabeledInput label="Number of test items" value={itemCount} onChangeText={setItemCount} />
            <LabeledInput label="Paper prompt variants (1-3)" value={variantCount} onChangeText={setVariantCount} />
            <Pressable onPress={() => setIncludeCorrectControl((v) => !v)} style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
              <View style={{ width: 20, height: 20, borderRadius: 4, borderWidth: 1, borderColor: "#64748b", backgroundColor: includeCorrectControl ? "#1d4ed8" : "white" }} />
              <Text>Run correct-suggestion control when baseline is wrong</Text>
            </Pressable>
            <Text style={{ color: "#6b7280", fontSize: 12 }}>
              Model calls: {includeCorrectControl ? `${counts.minimum} to ${counts.maximum}` : `${counts.minimum}`}. Each misleading condition is a two-turn conversation: baseline answer, then user pushback.
            </Text>
            <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap" }}>
              <Pressable onPress={previewPrompts} disabled={loading || previewLoading} style={{ flexGrow: 1, paddingVertical: 11, borderRadius: 8, alignItems: "center", borderWidth: 1, borderColor: "#1d4ed8" }}><Text style={{ color: "#1d4ed8", fontWeight: "700" }}>{previewLoading ? "Loading..." : "Preview conversations"}</Text></Pressable>
              <Pressable onPress={runBenchmark} disabled={loading || previewLoading} style={{ flexGrow: 1, paddingVertical: 11, borderRadius: 8, alignItems: "center", backgroundColor: loading ? "#93a4c7" : "#1d4ed8" }}><Text style={{ color: "white", fontWeight: "700" }}>{loading ? "Running..." : "Run benchmark"}</Text></Pressable>
            </View>
            {error ? <Text style={{ color: "#b91c1c" }}>{error}</Text> : null}
          </Section>

          {previewRows.length ? (
            <Section>
              <Text style={{ fontSize: 18, fontWeight: "700" }}>Prompt preview</Text>
              <Text style={{ color: "#6b7280", fontSize: 12 }}>All conversations are shown by default. Conditional correct-suggestion rows use a baseline-response placeholder.</Text>
              {previewRows.map((row) => (
                <View key={row.run_id} style={{ gap: 6, borderBottomWidth: 1, borderBottomColor: "#edf0f4", paddingBottom: 12 }}>
                  <Text style={{ fontWeight: "700" }}>{row.run_id}</Text>
                  <Text style={{ color: "#64748b", fontSize: 12 }}>{row.pressure_type ?? "baseline"}{row.conditional ? " · conditional" : ""}</Text>
                  {row.conditional_note ? <Text style={{ color: "#92400e", fontSize: 12 }}>{row.conditional_note}</Text> : null}
                  <Conversation messages={row.messages} />
                </View>
              ))}
            </Section>
          ) : null}

          {report ? (
            <>
              <Section>
                <Text style={{ fontSize: 18, fontWeight: "700" }}>SycoBench-600 metrics</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 10 }}>
                  <Card title="Acc" value={pct(report.baseline_accuracy)} note="Baseline accuracy" />
                  <Card title="PRA_all" value={pct(report.pressure_robust_accuracy)} note="Baseline correct + all 3 pressures correct" />
                  <Card title="Conditional robustness" value={pct(report.conditional_pressure_robustness)} note="All 3 pressures correct | baseline correct" />
                  <Card title="PRA_mean" value={pct(report.pressure_accuracy_mean)} note="Mean post-pressure accuracy" />
                  <Card title="Syco" value={pct(report.sycophancy_flip_to_wrong_rate)} note="Macro mean of Syco_t" />
                  <Card title="WrongFlip" value={pct(report.wrong_suggestion_flip_rate)} note="Explicit wrong-suggestion flip rate" />
                  <Card title="Update" value={pct(report.corrective_update_rate)} />
                  <Card title="Stub_nc" value={pct(report.stubbornness_rate)} />
                  <Card title="Sel" value={pct(report.correction_selectivity)} note="Update − WrongFlip" />
                </View>
              </Section>

              <Section>
                <Text style={{ fontSize: 18, fontWeight: "700" }}>Sycophancy by pressure type</Text>
                {Object.entries(report.sycophancy_by_pressure_type).map(([type, value]) => (
                  <View key={type} style={{ borderBottomWidth: 1, borderBottomColor: "#edf0f4", paddingVertical: 8 }}>
                    <Text style={{ fontWeight: "700" }}>{type.replaceAll("_", " ")}</Text>
                    <Text>Flip-to-wrong: {pct(value)}</Text>
                    <Text style={{ color: "#4b5563" }}>Post-pressure accuracy: {pct(report.pressure_accuracy_by_type[type])}</Text>
                    <Text style={{ color: "#4b5563" }}>Answer-change rate: {pct(report.answer_change_rate_by_type[type])}</Text>
                  </View>
                ))}
              </Section>

              <Section>
                <Text style={{ fontSize: 18, fontWeight: "700" }}>Additional diagnostics</Text>
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 10 }}>
                  <Card title="Exact wrong-option adoption" value={pct(report.regressive_sycophancy_rate)} note="Strict subset of WrongFlip" />
                  <Card title="Strict selectivity" value={pct(report.strict_correction_selectivity)} />
                  <Card title="Wrong-suggestion error delta" value={pp(report.mean_user_belief_error_delta)} />
                  <Card title="Pressure resistance" value={pct(report.pressure_resistance_rate)} />
                  <Card title="Parse compliance" value={pct(report.parse_compliance_rate)} />
                  <Card title="Exact one-letter compliance" value={pct(report.exact_one_letter_compliance_rate)} />
                </View>
              </Section>

              <Section>
                <Text style={{ fontSize: 18, fontWeight: "700" }}>Per-domain breakdown</Text>
                {Object.entries(report.domain_breakdown).map(([domain, values]) => (
                  <View key={domain} style={{ borderBottomWidth: 1, borderBottomColor: "#edf0f4", paddingVertical: 8 }}>
                    <Text style={{ fontWeight: "700" }}>{domain} · trials={values.trials}</Text>
                    <Text>Acc: {pct(values.baseline_accuracy)} · Syco: {pct(values.sycophancy_macro_rate)} · PRA_all: {pct(values.pressure_robust_accuracy)}</Text>
                    <Text style={{ color: "#4b5563" }}>WrongFlip: {pct(values.wrong_suggestion_flip_rate)} · Exact adoption: {pct(values.exact_wrong_suggestion_adoption_rate)}</Text>
                  </View>
                ))}
              </Section>

              <Section>
                <Text style={{ fontSize: 18, fontWeight: "700" }}>Classified runs and exact conversations</Text>
                <Text style={{ color: "#6b7280", fontSize: 12 }}>Prompts are displayed by default so each measurement can be audited.</Text>
                {runs.map((run) => (
                  <View key={run.run_id} style={{ paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: "#edf0f4", gap: 5 }}>
                    <Text style={{ fontWeight: "700" }}>{run.run_id}</Text>
                    <Text style={{ fontSize: 12, color: "#4b5563" }}>answer={run.answer ?? "unparsed"} · truth={run.correct_answer} · {run.pressure_type ?? "baseline"}{run.user_suggestion ? ` · user=${run.user_suggestion}` : ""}</Text>
                    <Text style={{ fontSize: 12, fontWeight: "600" }}>{run.classification}</Text>
                    <Conversation messages={run.messages} rawAnswer={run.raw_answer} />
                  </View>
                ))}
              </Section>
            </>
          ) : null}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
