// Test script for self-correction pattern detection
// Run: node test_self_correction.js

const CORRECTION_PATTERNS = [
  { pattern: /我之前说错了/, type: "self_correction" },
  { pattern: /更正[:：]\s*(.+)/, type: "explicit" },
  { pattern: /修正[:：]\s*(.+)/, type: "explicit" },
  { pattern: /纠正[:：]\s*(.+)/, type: "explicit" },
  { pattern: /不是(.{1,30})[，,](?:\s*)是(.+)/, type: "correction" },
  { pattern: /不是(.{1,30})[。.](?:\s*)是(.+)/, type: "correction" },
  { pattern: /i was wrong/i, type: "self_correction" },
  { pattern: /correction[:：]\s*(.+)/i, type: "explicit" },
];

function detectCorrections(text) {
  const str = String(text || "");
  if (!str.trim()) return [];
  const corrections = [];
  for (const { pattern, type } of CORRECTION_PATTERNS) {
    const match = str.match(pattern);
    if (match) {
      corrections.push({
        type,
        correctedContent: match[2] || match[1] || match[0],
        fullMatch: match[0],
      });
    }
  }
  return corrections;
}

function extractCorrectionQuery(assistantText, correction) {
  if (correction.type === "correction") {
    const negMatch = assistantText.match(/不是(.{1,30})[，,。.]/);
    if (negMatch) return negMatch[1].trim();
  }
  if (correction.type === "explicit") {
    const negMatch = assistantText.match(/不是(.{1,30})[，,。.]/);
    if (negMatch) return negMatch[1].trim();
    return correction.correctedContent;
  }
  const beforeMatch = assistantText.match(/之前.{0,20}(说|认为|提到)(.{0,60})/);
  if (beforeMatch) return beforeMatch[2].trim();
  return correction.correctedContent;
}

// ===== Test Cases =====
const tests = [
  {
    name: "Test 1: Explicit correction (更正)",
    input: "更正：MAGMA 做 L0 捕获，不是腾讯插件。",
    expectCorrection: true,
    expectQuery: "腾讯插件",  // Should extract the wrong part, not the whole sentence
  },
  {
    name: "Test 2: Self correction (我之前说错了)",
    input: "我之前说错了，GBrain 已经删除了。",
    expectCorrection: true,
    expectQuery: "GBrain",
  },
  {
    name: "Test 3: No correction",
    input: "MAGMA 是知识图谱系统。",
    expectCorrection: false,
  },
  {
    name: "Test 4: Negation correction (不是X，是Y)",
    input: "不是 26 个工具，是 20 个。",
    expectCorrection: true,
    expectQuery: "26 个工具",
  },
  {
    name: "Test 5: False positive check (normal statement)",
    input: "准确地说，这个功能已经在 v2 中实现了。",
    expectCorrection: false,
  },
  {
    name: "Test 6: False positive check (actually)",
    input: "Actually, the system supports multiple languages.",
    expectCorrection: false,
  },
];

let passed = 0;
let failed = 0;

for (const test of tests) {
  const corrections = detectCorrections(test.input);
  const hasCorrection = corrections.length > 0;

  if (hasCorrection !== test.expectCorrection) {
    console.log(`❌ ${test.name}: expected correction=${test.expectCorrection}, got ${hasCorrection}`);
    failed++;
    continue;
  }

  if (test.expectCorrection && test.expectQuery) {
    const query = extractCorrectionQuery(test.input, corrections[0]);
    if (!query.includes(test.expectQuery)) {
      console.log(`❌ ${test.name}: expected query to contain "${test.expectQuery}", got "${query}"`);
      failed++;
      continue;
    }
  }

  console.log(`✅ ${test.name}`);
  passed++;
}

console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
