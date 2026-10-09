/**
 * RFC 4180 Compliant Stateful Delimited Text Parser
 * Correctly handles:
 * - Multi-line reviews containing newlines (\n, \r\n) inside quotes
 * - Escaped quotes ("")
 * - Embedded delimiters (commas, tabs, semicolons) inside quotes
 * - Byte Order Marks (BOM)
 * - Automatic delimiter detection (, \t ; |)
 */

export function parseDelimitedText(text: string, defaultDelimiter?: string): string[][] {
  const rows: string[][] = [];
  if (!text || !text.trim()) return rows;

  // Strip BOM if present
  const cleanText = text.replace(/^\uFEFF/, '').trim();
  if (!cleanText) return rows;

  // Auto-detect delimiter from the first line if not provided
  let delimiter = defaultDelimiter;
  if (!delimiter) {
    const firstLineEnd = cleanText.search(/[\r\n]/);
    const firstLine = firstLineEnd !== -1 ? cleanText.slice(0, firstLineEnd) : cleanText;
    delimiter = firstLine.includes('\t')
      ? '\t'
      : firstLine.includes(';')
      ? ';'
      : firstLine.includes('|')
      ? '|'
      : ',';
  }

  let currentRow: string[] = [];
  let currentField = '';
  let inQuotes = false;
  const len = cleanText.length;

  for (let i = 0; i < len; i++) {
    const char = cleanText[i];
    const nextChar = i + 1 < len ? cleanText[i + 1] : '';

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        // Escaped quote: "" inside a quoted string
        currentField += '"';
        i++; // skip second quote
      } else {
        // Toggle quote state
        inQuotes = !inQuotes;
      }
    } else if (char === delimiter && !inQuotes) {
      // End of field
      currentRow.push(currentField.trim());
      currentField = '';
    } else if ((char === '\r' || char === '\n') && !inQuotes) {
      // End of line outside of quotes
      if (char === '\r' && nextChar === '\n') {
        i++; // skip \n
      }
      currentRow.push(currentField.trim());
      currentField = '';

      // Only push non-empty rows
      if (currentRow.some((f) => f.length > 0)) {
        rows.push(currentRow);
      }
      currentRow = [];
    } else {
      currentField += char;
    }
  }

  // Push final field and row
  if (currentField.length > 0 || currentRow.length > 0) {
    currentRow.push(currentField.trim());
    if (currentRow.some((f) => f.length > 0)) {
      rows.push(currentRow);
    }
  }

  return rows;
}
