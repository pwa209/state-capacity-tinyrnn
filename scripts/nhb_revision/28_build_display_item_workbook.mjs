import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "C:\\Users\\Gebruiker\\Documents\\TinyRNN State and Capacity\\state_capacity_tinyrnn";
const OUT_DIR = path.join(ROOT, "outputs", "nhb_revision", "display_item_revision");
const PAYLOAD = path.join(OUT_DIR, "display_item_workbook_payload.json");
const OUTPUT_XLSX = path.join(OUT_DIR, "state_capacity_NHB_NMI_revised_figure_table_source_data.xlsx");
const PREVIEW_DIR = path.join(OUT_DIR, "workbook_previews");

function asMatrix(rows) {
  const maxCols = Math.max(...rows.map((row) => row.length));
  return rows.map((row) => {
    const out = [...row];
    while (out.length < maxCols) out.push(null);
    return out;
  });
}

function safeSheetName(name, used) {
  let candidate = String(name).replace(/[\\/?*\[\]:]/g, "_").slice(0, 31);
  if (!candidate) candidate = "Sheet";
  let suffix = 1;
  const base = candidate.slice(0, 27);
  while (used.has(candidate)) {
    candidate = `${base}_${suffix}`;
    suffix += 1;
  }
  used.add(candidate);
  return candidate;
}

function applySheetStyle(sheet, rowCount, colCount) {
  if (rowCount < 1 || colCount < 1) return;
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  header.format = {
    fill: "#1F2937",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  const visibleCols = Math.min(colCount, 18);
  for (let c = 0; c < visibleCols; c += 1) {
    const range = sheet.getRangeByIndexes(0, c, Math.min(rowCount, 250), 1);
    range.format.columnWidth = c < 3 ? 22 : 18;
  }
  const firstRows = sheet.getRangeByIndexes(0, 0, Math.min(rowCount, 60), Math.min(colCount, 18));
  firstRows.format = {
    wrapText: true,
    verticalAlignment: "Top",
  };
}

async function main() {
  const payload = JSON.parse(await fs.readFile(PAYLOAD, "utf8"));
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });

  const workbook = Workbook.create();
  const usedNames = new Set();

  for (const sheetSpec of payload.sheets) {
    const sheetName = safeSheetName(sheetSpec.name, usedNames);
    const sheet = workbook.worksheets.add(sheetName);
    const rows = asMatrix(sheetSpec.rows);
    const rowCount = rows.length;
    const colCount = rows[0]?.length ?? 1;
    if (rowCount > 0 && colCount > 0) {
      sheet.getRangeByIndexes(0, 0, rowCount, colCount).values = rows;
      applySheetStyle(sheet, rowCount, colCount);
    }
  }

  const summary = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 6000,
  });
  console.log(summary.ndjson);

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 200 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);

  for (const sheetName of usedNames) {
    const preview = await workbook.render({
      sheetName,
      range: "A1:H20",
      scale: 1,
      format: "png",
    });
    const bytes = new Uint8Array(await preview.arrayBuffer());
    if (bytes.length === 0) {
      throw new Error(`Empty preview render for ${sheetName}`);
    }
    if (["INDEX", "FIGURE_PLAN", "CAPTIONS_REVISED"].includes(sheetName)) {
      await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName}.png`), bytes);
    }
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_XLSX);
  console.log(`Wrote ${OUTPUT_XLSX}`);
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
