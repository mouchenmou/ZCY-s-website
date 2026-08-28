const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const [inputArg, outputArg] = process.argv.slice(2);

if (!inputArg || !outputArg) {
  console.error("Usage: node scripts/md-to-pdf.js <input.md> <output.pdf>");
  process.exit(2);
}

const root = process.cwd();
const inputPath = path.resolve(root, inputArg);
const outputPath = path.resolve(root, outputArg);
const tmpDir = path.resolve(root, "tmp", "pdfs");
const htmlPath = path.join(tmpDir, `${path.basename(outputPath, ".pdf")}.html`);

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(value) {
  let out = escapeHtml(value);
  out = out.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
    const normalized = src.trim();
    return `<figure><img src="${escapeHtml(normalized)}" alt="${escapeHtml(alt)}"></figure>`;
  });
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return out;
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  let inCode = false;
  let code = [];
  let listItems = [];

  function flushParagraph() {
    if (paragraph.length) {
      blocks.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  }

  function flushList() {
    if (listItems.length) {
      blocks.push(`<ul>${listItems.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`);
      listItems = [];
    }
  }

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      flushParagraph();
      flushList();
      if (inCode) {
        blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = [];
      }
      inCode = !inCode;
      continue;
    }

    if (inCode) {
      code.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      flushParagraph();
      flushList();
      blocks.push("<hr>");
      continue;
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = /^[-*+]\s+(.+)$/.exec(trimmed);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      continue;
    }

    const quote = /^>\s?(.*)$/.exec(trimmed);
    if (quote) {
      flushParagraph();
      flushList();
      blocks.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();

  return blocks.join("\n");
}

function fileUri(filePath) {
  return `file:///${filePath.replace(/\\/g, "/").split("/").map(encodeURIComponent).join("/")}`;
}

const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];

const browserPath = chromeCandidates.find((candidate) => fs.existsSync(candidate));
if (!browserPath) {
  console.error("Could not find Chrome or Edge.");
  process.exit(1);
}

const markdown = fs.readFileSync(inputPath, "utf8");
const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <base href="${fileUri(path.dirname(inputPath))}/">
  <title>${escapeHtml(path.basename(inputPath))}</title>
  <style>
    @page { size: Letter; margin: 0.72in 0.76in; }
    * { box-sizing: border-box; }
    body {
      color: #1f2933;
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 11.5pt;
      line-height: 1.5;
      margin: 0;
    }
    h1, h2, h3, h4 {
      color: #111827;
      line-height: 1.22;
      margin: 1.15em 0 0.45em;
      page-break-after: avoid;
    }
    h1 { font-size: 21pt; border-bottom: 1px solid #d5dbe3; padding-bottom: 0.22em; }
    h2 { font-size: 17pt; }
    h3 { font-size: 13.5pt; }
    p { margin: 0 0 0.78em; }
    blockquote {
      border-left: 3px solid #9aa6b2;
      color: #384452;
      margin: 0.8em 0;
      padding: 0.05em 0 0.05em 0.85em;
    }
    code {
      background: #f3f5f7;
      border-radius: 4px;
      font-family: Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
      padding: 0.08em 0.24em;
    }
    pre {
      background: #f6f8fa;
      border: 1px solid #d8dee4;
      border-radius: 6px;
      overflow: hidden;
      padding: 0.8em;
      white-space: pre-wrap;
    }
    pre code { background: transparent; padding: 0; }
    hr { border: 0; border-top: 1px solid #d5dbe3; margin: 1.1em 0; }
    figure { margin: 0.7em 0 1em; page-break-inside: avoid; }
    img { display: block; max-width: 100%; height: auto; }
    ul { margin: 0.2em 0 0.8em 1.25em; padding: 0; }
  </style>
</head>
<body>
${renderMarkdown(markdown)}
</body>
</html>`;

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.mkdirSync(tmpDir, { recursive: true });
fs.writeFileSync(htmlPath, html, "utf8");

const result = spawnSync(browserPath, [
  "--headless=new",
  "--disable-gpu",
  "--no-pdf-header-footer",
  `--print-to-pdf=${outputPath}`,
  fileUri(htmlPath),
], { encoding: "utf8" });

if (result.status !== 0) {
  process.stderr.write(result.stderr || result.stdout || "PDF export failed.\n");
  process.exit(result.status || 1);
}

console.log(`Wrote ${outputPath}`);
