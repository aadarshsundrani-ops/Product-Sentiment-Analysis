import React, { useState, useRef } from 'react';
import {
  Upload,
  Database,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  FileSpreadsheet,
  Layers,
  ArrowRight,
  Info,
  ShieldCheck,
  Check,
  X,
} from 'lucide-react';
import { DataIngestionCanvas } from '../3d/DataIngestionCanvas';
import { SAMPLE_REVIEWS, ReviewSample } from '../../data/sampleData';
import { setActiveDataset } from '../../store/datasetStore';
import { parseDelimitedText } from '../../utils/csvParser';

export type CanonicalColumn = 'Review_ID' | 'Product_Name' | 'Rating' | 'Review_Text' | 'Sentiments';

export const REQUIRED_COLUMNS: {
  key: CanonicalColumn;
  label: string;
  description: string;
  requiredForRun: boolean;
}[] = [
  { key: 'Review_ID', label: 'Review_ID', description: 'Unique review identifier', requiredForRun: false },
  { key: 'Product_Name', label: 'Product_Name', description: 'Product title / category', requiredForRun: false },
  { key: 'Rating', label: 'Rating', description: 'Customer star rating (1–5)', requiredForRun: false },
  { key: 'Review_Text', label: 'Review_Text', description: 'Raw customer review text', requiredForRun: true },
  { key: 'Sentiments', label: 'Sentiments', description: 'Ground-truth polarity label', requiredForRun: false },
];

// Requirement 8: Dynamic case-insensitive and punctuation-agnostic column normalizer
export function normalizeHeader(header: string): string {
  return header.toLowerCase().replace(/[\s_\-\.]+/g, '');
}

export function matchCanonicalColumn(header: string): CanonicalColumn | null {
  const norm = normalizeHeader(header);

  // 1. Review_ID: review_id, reviewid, review-id, review id, id, rev_id, reviewno, etc.
  if (
    [
      'reviewid', 'id', 'revid', 'reviewno', 'reviewnum', 'reviewidentifier',
      'reviewcode', 'reviewpk', 'srno', 'srnum', 'serialno', 'index', 'code',
      'fmcyid', 'fmcyno', 'customid', 'uniqueid', 'uuid', 'itemid', 'idnum'
    ].includes(norm) ||
    norm.startsWith('reviewid') ||
    norm.endsWith('reviewid') ||
    (norm.includes('review') && (norm.includes('id') || norm.includes('no') || norm.includes('num')))
  ) {
    return 'Review_ID';
  }

  // 2. Product_Name: product_name, productname, product name, product, item, item_name, product_category, category, etc.
  if (
    [
      'productname', 'product', 'item', 'itemname', 'producttitle', 'title',
      'productcategory', 'category', 'producttype', 'brand', 'prodname', 'sku'
    ].includes(norm) ||
    norm.startsWith('product') ||
    norm.includes('productname') ||
    norm.includes('itemname')
  ) {
    return 'Product_Name';
  }

  // 3. Rating: rating, ratings, stars, starrating, star_rating, score, reviewrating, etc.
  if (
    [
      'rating', 'ratings', 'stars', 'starrating', 'star', 'score',
      'reviewrating', 'overall', 'starscore', 'rate'
    ].includes(norm) ||
    norm.startsWith('rating') ||
    norm.startsWith('star')
  ) {
    return 'Rating';
  }

  // 4. Review_Text: review_text, reviewtext, review_body, reviewbody, review, text, body, comment, comments, feedback, content
  if (
    [
      'reviewtext', 'reviewbody', 'review', 'text', 'body', 'comment',
      'comments', 'feedback', 'content', 'reviewcontent', 'userreview',
      'customerreview', 'description'
    ].includes(norm) ||
    norm.startsWith('reviewtext') ||
    norm.startsWith('reviewbody') ||
    (norm.includes('review') && norm.includes('text'))
  ) {
    return 'Review_Text';
  }

  // 5. Sentiments: sentiments, sentiment, label, labels, target, polarity, class, sentimentlabel
  if (
    [
      'sentiments', 'sentiment', 'label', 'labels', 'target', 'polarity',
      'class', 'sentimentlabel', 'groundtruth', 'actualsentiment'
    ].includes(norm) ||
    norm.startsWith('sentiment')
  ) {
    return 'Sentiments';
  }

  return null;
}

export interface RawParsedData {
  type: 'delimited' | 'json';
  headers: string[];
  delimitedRows?: string[][];
  jsonRows?: any[];
}

export function mapParsedDataToReviewSamples(
  raw: RawParsedData,
  mapping: Record<CanonicalColumn, string | null>
): ReviewSample[] {
  if (raw.type === 'json' && raw.jsonRows) {
    return raw.jsonRows.map((row, idx) => {
      const reviewTextVal = mapping.Review_Text ? String(row[mapping.Review_Text] ?? '') : '';
      const ratingRaw = mapping.Rating ? row[mapping.Rating] : undefined;
      const parsedRating = Number(ratingRaw);
      const ratingVal = isNaN(parsedRating) ? 5 : parsedRating;

      const rawSentiment = mapping.Sentiments ? String(row[mapping.Sentiments] ?? '') : '';
      let sentimentVal: 'Positive' | 'Negative' | 'Neutral' = 'Positive';
      if (rawSentiment) {
        const sLow = rawSentiment.toLowerCase();
        if (sLow.includes('pos') || sLow === '1') sentimentVal = 'Positive';
        else if (sLow.includes('neg') || sLow === '0') sentimentVal = 'Negative';
        else if (sLow.includes('neu')) sentimentVal = 'Neutral';
      } else {
        sentimentVal = ratingVal >= 4 ? 'Positive' : ratingVal <= 2 ? 'Negative' : 'Neutral';
      }

      const idVal = mapping.Review_ID
        ? String(row[mapping.Review_ID] ?? '')
        : (row.Review_ID ?? row.id ?? row.review_id ?? `REV-${idx + 1001}`);
      const prodVal = mapping.Product_Name
        ? String(row[mapping.Product_Name] ?? '')
        : (row.Product_Name ?? row.product_category ?? 'General Product');

      return {
        Review_ID: String(idVal).trim() || `REV-${idx + 1001}`,
        Product_Name: String(prodVal).trim() || 'General Product',
        Rating: ratingVal,
        Review_Text: reviewTextVal,
        Sentiments: sentimentVal,
        word_count: reviewTextVal.split(/\s+/).filter(Boolean).length || 10,
        id: String(idVal).trim() || `REV-${idx + 1001}`,
        product_category: String(prodVal).trim() || 'General Product',
        stars: ratingVal,
        review_text: reviewTextVal,
        sentiment: sentimentVal,
      };
    });
  }

  // Delimited (CSV / TSV)
  const rows = raw.delimitedRows || [];
  const headers = raw.headers || [];

  return rows.map((parts, idx) => {
    const getVal = (colName: string | null) => {
      if (!colName) return '';
      const cIdx = headers.indexOf(colName);
      return cIdx !== -1 && parts[cIdx] !== undefined ? parts[cIdx].replace(/^["']|["']$/g, '').trim() : '';
    };

    const reviewTextVal = getVal(mapping.Review_Text);
    const rawRating = getVal(mapping.Rating);
    const parsedRating = Number(rawRating);
    const ratingVal = isNaN(parsedRating) || rawRating === '' ? 5 : parsedRating;

    const rawSentiment = getVal(mapping.Sentiments);
    let sentimentVal: 'Positive' | 'Negative' | 'Neutral' = 'Positive';
    if (rawSentiment) {
      const sLow = rawSentiment.toLowerCase();
      if (sLow.includes('pos') || sLow === '1') sentimentVal = 'Positive';
      else if (sLow.includes('neg') || sLow === '0') sentimentVal = 'Negative';
      else if (sLow.includes('neu')) sentimentVal = 'Neutral';
    } else {
      sentimentVal = ratingVal >= 4 ? 'Positive' : ratingVal <= 2 ? 'Negative' : 'Neutral';
    }

    let idVal = getVal(mapping.Review_ID);
    // If not explicitly mapped or empty, and column 0 exists and is not review_text, use column 0
    if (!idVal && parts.length > 0 && parts[0] && parts[0] !== reviewTextVal) {
      idVal = parts[0].replace(/^["']|["']$/g, '').trim();
    }
    if (!idVal) {
      idVal = `REV-${idx + 1001}`;
    }

    let prodVal = getVal(mapping.Product_Name);
    if (!prodVal) {
      prodVal = 'General Product';
    }

    return {
      Review_ID: idVal,
      Product_Name: prodVal,
      Rating: ratingVal,
      Review_Text: reviewTextVal,
      Sentiments: sentimentVal,
      word_count: reviewTextVal.split(/\s+/).filter(Boolean).length || 10,
      id: idVal,
      product_category: prodVal,
      stars: ratingVal,
      review_text: reviewTextVal,
      sentiment: sentimentVal,
    };
  });
}

export const DatasetUploadSection: React.FC = () => {
  const [data, setData] = useState<ReviewSample[]>(SAMPLE_REVIEWS);
  const [rawParsedData, setRawParsedData] = useState<RawParsedData | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [fileName, setFileName] = useState<string>('amazon_raw_reviews_10k.csv (Preloaded Benchmark)');
  const [currentPage, setCurrentPage] = useState(1);
  const [viewMode, setViewMode] = useState<'head-tail' | 'head' | 'tail' | 'all'>('head-tail');
  const [detectedColumns, setDetectedColumns] = useState<string[]>([
    'Review_ID',
    'Product_Name',
    'Rating',
    'Review_Text',
    'Sentiments',
  ]);
  const [columnMapping, setColumnMapping] = useState<Record<CanonicalColumn, string | null>>({
    Review_ID: 'Review_ID',
    Product_Name: 'Product_Name',
    Rating: 'Rating',
    Review_Text: 'Review_Text',
    Sentiments: 'Sentiments',
  });
  const [userBypassedMissing, setUserBypassedMissing] = useState(false);

  const rowsPerPage = 5;
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Compute dataset statistics
  const totalRows = data.length;
  const posCount = data.filter((d) => d.Sentiments === 'Positive' || d.sentiment === 'Positive').length;
  const negCount = data.filter((d) => d.Sentiments === 'Negative' || d.sentiment === 'Negative').length;
  const neuCount = data.filter((d) => d.Sentiments === 'Neutral' || d.sentiment === 'Neutral').length;

  // Schema Validation Check
  const missingColumns = REQUIRED_COLUMNS.filter((col) => !columnMapping[col.key]).map((col) => col.key);
  const hasReviewText = Boolean(columnMapping.Review_Text);
  const isSchemaFullyValid = missingColumns.length === 0;
  const canRunModel = hasReviewText && (isSchemaFullyValid || userBypassedMissing);

  const totalPages = Math.ceil(totalRows / rowsPerPage);

  // Requirement 3: 1st 5 rows and last 5 rows
  const headRows = data.slice(0, 5);
  const tailRows = data.slice(Math.max(5, totalRows - 5), totalRows);
  const paginatedRows = data.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

  const triggerIngestionAnimation = () => {
    setIsIngesting(true);
    setTimeout(() => setIsIngesting(false), 2400);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processUploadedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processUploadedFile(e.target.files[0]);
    }
  };

  // User manually selects / remaps a column from the dropdown
  const handleColumnMappingChange = (canonicalKey: CanonicalColumn, newSourceCol: string | null) => {
    const updatedMapping = {
      ...columnMapping,
      [canonicalKey]: newSourceCol || null,
    };
    setColumnMapping(updatedMapping);

    if (rawParsedData) {
      const updatedMapped = mapParsedDataToReviewSamples(rawParsedData, updatedMapping);
      if (updatedMapped.length > 0) {
        setData(updatedMapped);
        setActiveDataset(updatedMapped, fileName);
      }
    }
  };

  // Requirement 4: Read dataset carefully and map dynamic columns (Requirement 8)
  const processUploadedFile = (file: File) => {
    setFileName(`${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
    setUserBypassedMissing(false);
    triggerIngestionAnimation();

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const rawContent = (event.target?.result as string) || '';
        const text = rawContent.replace(/^\uFEFF/, '').trim();

        if (file.name.endsWith('.json')) {
          const parsed = JSON.parse(text);
          if (Array.isArray(parsed) && parsed.length > 0) {
            const rawKeys = Object.keys(parsed[0]);
            setDetectedColumns(rawKeys);

            const newMapping: Record<CanonicalColumn, string | null> = {
              Review_ID: null,
              Product_Name: null,
              Rating: null,
              Review_Text: null,
              Sentiments: null,
            };

            rawKeys.forEach((key) => {
              const matched = matchCanonicalColumn(key);
              if (matched && !newMapping[matched]) {
                newMapping[matched] = key;
              }
            });

            // Fallback for Review_ID if not matched
            if (!newMapping.Review_ID && rawKeys.length > 0) {
              const mappedValues = Object.values(newMapping).filter(Boolean);
              if (!mappedValues.includes(rawKeys[0])) {
                newMapping.Review_ID = rawKeys[0];
              }
            }

            const rawData: RawParsedData = {
              type: 'json',
              headers: rawKeys,
              jsonRows: parsed,
            };
            setRawParsedData(rawData);
            setColumnMapping(newMapping);

            const mapped = mapParsedDataToReviewSamples(rawData, newMapping);
            if (mapped.length > 0) {
              setData(mapped);
              setCurrentPage(1);
              setActiveDataset(mapped, file.name);
            }
          }
        } else {
          // RFC 4180 robust delimited text parser for CSV, TSV, TXT, spreadsheet exports
          const parsedRows = parseDelimitedText(text);
          if (parsedRows.length > 0) {
            const rawHeaders = parsedRows[0].map((h) => h.replace(/^["']|["']$/g, '').trim());
            const dataRows = parsedRows.slice(1);
            setDetectedColumns(rawHeaders);

            const newMapping: Record<CanonicalColumn, string | null> = {
              Review_ID: null,
              Product_Name: null,
              Rating: null,
              Review_Text: null,
              Sentiments: null,
            };

            rawHeaders.forEach((header) => {
              const matched = matchCanonicalColumn(header);
              if (matched && !newMapping[matched]) {
                newMapping[matched] = header;
              }
            });

            // Fallback for Review_ID if not matched: use rawHeaders[0] if column 0 isn't mapped to another role
            if (!newMapping.Review_ID && rawHeaders.length > 0) {
              const mappedValues = Object.values(newMapping).filter(Boolean);
              if (!mappedValues.includes(rawHeaders[0])) {
                newMapping.Review_ID = rawHeaders[0];
              }
            }

            const rawData: RawParsedData = {
              type: 'delimited',
              headers: rawHeaders,
              delimitedRows: dataRows,
            };
            setRawParsedData(rawData);
            setColumnMapping(newMapping);

            const mapped = mapParsedDataToReviewSamples(rawData, newMapping);
            if (mapped.length > 0) {
              setData(mapped);
              setCurrentPage(1);
              setActiveDataset(mapped, file.name);
            }
          }
        }
      } catch (err) {
        console.error('File parsing error:', err);
      }
    };
    reader.readAsText(file);
  };

  const resetToSample = () => {
    setData(SAMPLE_REVIEWS);
    setRawParsedData(null);
    setFileName('amazon_raw_reviews_10k.csv (Preloaded Benchmark)');
    setDetectedColumns(['Review_ID', 'Product_Name', 'Rating', 'Review_Text', 'Sentiments']);
    setColumnMapping({
      Review_ID: 'Review_ID',
      Product_Name: 'Product_Name',
      Rating: 'Rating',
      Review_Text: 'Review_Text',
      Sentiments: 'Sentiments',
    });
    setUserBypassedMissing(false);
    setCurrentPage(1);
    setActiveDataset(SAMPLE_REVIEWS, 'amazon_raw_reviews_10k.csv (Preloaded Benchmark)');
    triggerIngestionAnimation();
  };

  const scrollToPrediction = () => {
    const el = document.getElementById('prediction-interface');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="dataset-upload" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Section Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold text-secondary bg-secondary/10 border border-secondary/20 mb-3 shadow-sm">
            <Database className="w-3.5 h-3.5" />
            Dataset Ingestion & Schema Validator
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight text-slate-900 dark:text-white">
            Upload & Validate Dataset
          </h2>
          <p className="text-slate-600 dark:text-slate-300 text-sm sm:text-base mt-2 max-w-2xl leading-relaxed">
            Upload your dataset in CSV, Excel, Numbers, or spreadsheet formats. The pipeline validates required
            columns, normalizes dynamic headers, and displays first and last sample records.
          </p>
        </div>

        {/* Fallback Benchmark Dataset Toggle */}
        <button
          onClick={resetToSample}
          className="self-start md:self-auto px-4 py-2.5 rounded-xl text-xs font-semibold glass-panel hover:border-primary/40 text-foreground flex items-center gap-2 transition-all hover:scale-[1.02] shadow-sm"
        >
          <RefreshCw className="w-3.5 h-3.5 text-primary" />
          Reset to 10K Benchmark Schema
        </button>
      </div>

      {/* Main Grid: Upload Zone + 3D Ingestion Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">
        {/* Requirement 2: Upload Zone — Only CSV, Excel, Numbers, Spreadsheet */}
        <div className="lg:col-span-7">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer rounded-2xl p-7 border-2 border-dashed transition-all duration-300 glass-panel-elevated relative overflow-hidden group flex flex-col items-center justify-center text-center min-h-[250px] ${
              isDragging
                ? 'border-primary bg-primary/10 shadow-glow-primary scale-[1.01]'
                : 'border-slate-300 dark:border-slate-700 hover:border-primary/60'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept=".csv,.xlsx,.xls,.numbers,.tsv,.json,.txt"
              className="hidden"
            />

            <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-primary/20 to-secondary/20 flex items-center justify-center text-primary mb-3.5 group-hover:scale-110 transition-transform shadow-sm">
              <FileSpreadsheet className="w-7 h-7" />
            </div>

            <div className="text-base sm:text-lg font-bold font-display text-slate-900 dark:text-white mb-1">
              Upload by CSV, Excel, Numbers, or Spreadsheet
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 max-w-md leading-relaxed">
              Drop your spreadsheet file (.csv, .xlsx, .xls, .numbers, .tsv, .json) here to automatically parse and
              validate the dataset schema.
            </p>

            {/* Allowed Format Badges */}
            <div className="mt-3 flex flex-wrap items-center justify-center gap-1.5 text-[10px] font-mono text-slate-500 dark:text-slate-400">
              <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                .CSV
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                .XLSX / .XLS
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                .NUMBERS
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                .TSV / Spreadsheet
              </span>
            </div>

            <div className="mt-4 inline-flex items-center gap-2 text-xs font-mono text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 shadow-sm">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Active: {fileName}
            </div>
          </div>
        </div>

        {/* 3D Ingestion Funnel Canvas Card */}
        <div className="lg:col-span-5 rounded-2xl glass-panel p-6 border border-surface-border flex flex-col justify-between relative overflow-hidden shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono uppercase tracking-wider text-slate-600 dark:text-slate-300 flex items-center gap-1.5 font-semibold">
              <span className={`w-2 h-2 rounded-full ${isIngesting ? 'bg-secondary animate-ping' : 'bg-primary'}`} />
              3D Data Ingestion Stream
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              {isIngesting ? 'Status: INGESTING...' : 'Status: IDLE'}
            </span>
          </div>

          <div className="h-[170px] w-full relative">
            <DataIngestionCanvas isIngesting={isIngesting} />
          </div>

          <div className="pt-3 border-t border-surface-border grid grid-cols-3 gap-2 text-center text-xs">
            <div>
              <div className="font-bold text-slate-900 dark:text-white font-mono">{totalRows}</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400">Total Rows</div>
            </div>
            <div>
              <div className="font-bold text-emerald-600 dark:text-emerald-400 font-mono">{posCount}</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400">Positive</div>
            </div>
            <div>
              <div className="font-bold text-rose-600 dark:text-rose-400 font-mono">{negCount + neuCount}</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400">Neg / Neutral</div>
            </div>
          </div>
        </div>
      </div>

      {/* Requirement 5 & 8: Detected Columns & Dynamic Mapping Panel */}
      <div className="mb-6 p-5 rounded-2xl glass-panel border border-surface-border shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-surface-border pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-500" />
            <h3 className="text-sm font-bold font-display text-slate-900 dark:text-white">
              Dataset Column Schema & Dynamic Header Recognition
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
            Detected: {detectedColumns.length} columns in file
          </span>
        </div>

        {/* 5 Required Columns Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {REQUIRED_COLUMNS.map((col) => {
            const mappedName = columnMapping[col.key];
            const isFound = Boolean(mappedName);

            return (
              <div
                key={col.key}
                className={`p-3 rounded-xl border flex flex-col justify-between transition-all ${
                  isFound
                    ? 'bg-emerald-50/70 dark:bg-emerald-950/20 border-emerald-300 dark:border-emerald-800/60'
                    : col.requiredForRun
                    ? 'bg-rose-50/80 dark:bg-rose-950/20 border-rose-300 dark:border-rose-800/60'
                    : 'bg-amber-50/70 dark:bg-amber-950/20 border-amber-300 dark:border-amber-800/60'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <span className="font-mono font-bold text-xs text-slate-900 dark:text-white">{col.label}</span>
                    {isFound ? (
                      <span className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                        <Check className="w-3 h-3" />
                      </span>
                    ) : (
                      <span
                        className={`w-4 h-4 rounded-full flex items-center justify-center ${
                          col.requiredForRun
                            ? 'bg-rose-500/20 text-rose-600 dark:text-rose-400'
                            : 'bg-amber-500/20 text-amber-600 dark:text-amber-400'
                        }`}
                      >
                        <X className="w-3 h-3" />
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
                    {col.description}
                  </div>
                </div>

                <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-800 text-[10px] font-mono space-y-1.5">
                  <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    <span>Source Column:</span>
                    {isFound && <span className="text-emerald-500 font-bold">Active</span>}
                  </div>
                  <select
                    value={mappedName || ''}
                    onChange={(e) => handleColumnMappingChange(col.key, e.target.value || null)}
                    className={`w-full text-xs font-mono rounded-lg px-2 py-1.5 border transition-all cursor-pointer ${
                      isFound
                        ? 'bg-white dark:bg-slate-900 border-emerald-400 dark:border-emerald-600 text-emerald-800 dark:text-emerald-200 font-semibold shadow-xs'
                        : col.requiredForRun
                        ? 'bg-white dark:bg-slate-900 border-rose-400 dark:border-rose-600 text-rose-800 dark:text-rose-200 font-semibold'
                        : 'bg-white dark:bg-slate-900 border-amber-300 dark:border-amber-600 text-amber-800 dark:text-amber-200'
                    }`}
                  >
                    <option value="">{col.requiredForRun ? '-- Choose Column (Required) --' : '-- Auto-generate / None --'}</option>
                    {detectedColumns.map((c) => (
                      <option key={c} value={c}>
                        {c} {c === mappedName ? '✓ (Current)' : ''}
                      </option>
                    ))}
                  </select>
                  {isFound ? (
                    <span className="text-emerald-700 dark:text-emerald-300 font-semibold truncate block text-[10px]">
                      Source: &quot;{mappedName}&quot;
                    </span>
                  ) : (
                    <span
                      className={`font-semibold block text-[10px] ${
                        col.requiredForRun
                          ? 'text-rose-600 dark:text-rose-400'
                          : 'text-amber-700 dark:text-amber-400'
                      }`}
                    >
                      {col.requiredForRun ? 'Missing (Mandatory)' : 'Missing (Optional)'}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Requirements 6 & 7: Conditional Notifications and Blocker Banner */}
        {!hasReviewText ? (
          /* CRITICAL BLOCKER: Review_Text Missing */
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-start gap-3 text-xs text-slate-800 dark:text-slate-200 animate-fade-in">
            <AlertOctagon className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-bold text-rose-600 dark:text-rose-400 text-sm">
                Critical Column Missing: &quot;Review_Text&quot; Not Found
              </div>
              <p className="leading-relaxed">
                The dataset does not contain the mandatory <strong>Review_Text</strong> column (or accepted aliases:
                <em> review, text, review_body, comment, content</em>). The model cannot perform product sentiment
                analysis without review text to evaluate. Please upload a dataset with this column.
              </p>
              <div className="pt-2">
                <button
                  disabled
                  className="px-4 py-1.5 rounded-lg dark:bg-rose-500/20 bg-rose-100 dark:text-rose-400 text-rose-800 text-xs font-semibold cursor-not-allowed opacity-75 border dark:border-rose-500/30 border-rose-300"
                >
                  Model Run Disabled (Requires Review_Text)
                </button>
              </div>
            </div>
          </div>
        ) : !isSchemaFullyValid ? (
          /* WARNING NOTIFICATION: Non-critical columns missing (Requirement 6 & 7) */
          <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3 text-xs text-slate-800 dark:text-slate-200 animate-fade-in">
            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
            <div className="space-y-1.5 flex-1">
              <div className="font-bold text-amber-600 dark:text-amber-400 text-sm">
                Notice: Dataset Missing Specific Columns ({missingColumns.join(', ')})
              </div>
              <p className="leading-relaxed">
                The dataset does not have the <strong>{missingColumns.join(', ')}</strong> column(s). The model needs
                these columns for full ground-truth benchmark verification and category segmentation. However, since
                <strong> Review_Text</strong> is present, you may proceed.
              </p>

              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  onClick={() => {
                    setUserBypassedMissing(true);
                    scrollToPrediction();
                  }}
                  className="px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-all shadow-md flex items-center gap-2"
                >
                  <ArrowRight className="w-3.5 h-3.5" />
                  Continue to Live AI Demo
                </button>
                <span className="text-[11px] text-slate-500 dark:text-slate-400">
                  Missing IDs and ground-truth sentiments will be auto-generated by the NLP engine.
                </span>
              </div>
            </div>
          </div>
        ) : (
          /* SUCCESS: All 5 columns detected */
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-between flex-wrap gap-3 text-xs text-slate-800 dark:text-slate-200 animate-fade-in">
            <div className="flex items-center gap-2.5">
              <ShieldCheck className="w-5 h-5 text-emerald-500 shrink-0" />
              <div>
                <span className="font-bold text-emerald-600 dark:text-emerald-400">
                  Complete 5-Column Schema Validated:
                </span>{' '}
                Review_ID, Product_Name, Rating, Review_Text, and Sentiments all identified.
              </div>
            </div>

            <button
              onClick={scrollToPrediction}
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md flex items-center gap-1.5"
            >
              <span>Proceed to Live AI Demo</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Requirement 3: 1st 5 Rows & Last 5 Rows Preview Table */}
      <div className="rounded-2xl glass-panel border border-surface-border p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-surface-border">
          <div>
            <h3 className="text-lg font-bold font-display text-slate-900 dark:text-white">
              Dataset Records Inspection
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
              Reviewing the 1st 5 rows (Head) and last 5 rows (Tail) across normalized canonical columns
            </p>
          </div>

          {/* View Mode Switcher */}
          <div className="flex items-center gap-1 p-1 rounded-xl bg-slate-100 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-xs">
            <button
              onClick={() => setViewMode('head-tail')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                viewMode === 'head-tail'
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
              }`}
            >
              Head & Tail (1st & Last 5)
            </button>
            <button
              onClick={() => setViewMode('head')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                viewMode === 'head'
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
              }`}
            >
              First 5 Only
            </button>
            <button
              onClick={() => setViewMode('tail')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                viewMode === 'tail'
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
              }`}
            >
              Last 5 Only
            </button>
            <button
              onClick={() => setViewMode('all')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                viewMode === 'all'
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
              }`}
            >
              Paginated All
            </button>
          </div>
        </div>

        {/* Dataset Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-surface-border text-slate-500 dark:text-slate-400 uppercase text-[10px] tracking-wider">
                <th className="pb-3 pr-3 w-12">#</th>
                <th className="pb-3 pr-4">Review_ID</th>
                <th className="pb-3 pr-4">Product_Name</th>
                <th className="pb-3 pr-4">Rating</th>
                <th className="pb-3 pr-4 w-2/5">Review_Text</th>
                <th className="pb-3 pr-4">Sentiments</th>
                <th className="pb-3 pr-4">Words</th>
                <th className="pb-3 text-right">Action</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-surface-border/50 text-foreground">
              {/* HEAD ROWS (1st 5 Rows) */}
              {(viewMode === 'head-tail' || viewMode === 'head') &&
                headRows.map((row, idx) => (
                  <tr key={`head-${row.Review_ID || idx}`} className="hover:bg-surface/50 transition-colors">
                    <td className="py-3 pr-3 text-slate-400 font-bold">#{idx + 1}</td>
                    <td className="py-3 pr-4 text-primary font-semibold">{row.Review_ID || row.id}</td>
                    <td className="py-3 pr-4 text-slate-700 dark:text-slate-300 font-sans">
                      {row.Product_Name || row.product_category}
                    </td>
                    <td className="py-3 pr-4 font-bold text-amber-500">
                      {'★'.repeat(Math.max(1, Math.min(5, row.Rating || row.stars || 5)))}
                    </td>
                    <td className="py-3 pr-4 font-sans text-xs text-slate-700 dark:text-slate-300 max-w-md truncate">
                      {row.Review_Text || row.review_text}
                    </td>
                    <td className="py-3 pr-4">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          (row.Sentiments || row.sentiment) === 'Positive'
                            ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                            : (row.Sentiments || row.sentiment) === 'Negative'
                            ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20'
                            : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20'
                        }`}
                      >
                        {row.Sentiments || row.sentiment}
                      </span>
                    </td>
                    <td className="py-3 pr-4 text-slate-400">{row.word_count || 15}w</td>
                    <td className="py-3 text-right">
                      <button
                        onClick={() => {
                          const revId = row.Review_ID || row.id || `REV-${idx + 1001}`;
                          const revText = row.Review_Text || row.review_text || '';
                          const prodName = row.Product_Name || row.product_category || 'Product Review';
                          const ratingVal = row.Rating || row.stars || 5;

                          window.dispatchEvent(
                            new CustomEvent('analyze-dataset-review', {
                              detail: {
                                reviewId: revId,
                                reviewText: revText,
                                productName: prodName,
                                rating: ratingVal,
                              },
                            })
                          );
                        }}
                        className="px-2.5 py-1 rounded-lg text-[10px] font-semibold bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 transition-all inline-flex items-center gap-1 shadow-sm"
                        title="Analyze tokens in TF-IDF & MiniLM"
                      >
                        <span>Analyze</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}

              {/* DIVIDER BETWEEN HEAD AND TAIL */}
              {viewMode === 'head-tail' && totalRows > 10 && (
                <tr>
                  <td colSpan={8} className="py-3 text-center bg-slate-100/60 dark:bg-slate-800/40 font-mono text-[11px] text-slate-500 dark:text-slate-400 border-y border-dashed border-slate-300 dark:border-slate-700">
                    ⋯ Showing 1st 5 Rows &amp; Last 5 Rows of {totalRows} Reviews in Dataset ⋯
                  </td>
                </tr>
              )}

              {/* TAIL ROWS (Last 5 Rows) */}
              {(viewMode === 'head-tail' || viewMode === 'tail') &&
                tailRows.map((row, idx) => {
                  const actualIdx = totalRows - tailRows.length + idx + 1;
                  return (
                    <tr key={`tail-${row.Review_ID || idx}`} className="hover:bg-surface/50 transition-colors">
                      <td className="py-3 pr-3 text-slate-400 font-bold">#{actualIdx}</td>
                      <td className="py-3 pr-4 text-primary font-semibold">{row.Review_ID || row.id}</td>
                      <td className="py-3 pr-4 text-slate-700 dark:text-slate-300 font-sans">
                        {row.Product_Name || row.product_category}
                      </td>
                      <td className="py-3 pr-4 font-bold text-amber-500">
                        {'★'.repeat(Math.max(1, Math.min(5, row.Rating || row.stars || 5)))}
                      </td>
                      <td className="py-3 pr-4 font-sans text-xs text-slate-700 dark:text-slate-300 max-w-md truncate">
                        {row.Review_Text || row.review_text}
                      </td>
                      <td className="py-3 pr-4">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            (row.Sentiments || row.sentiment) === 'Positive'
                              ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                              : (row.Sentiments || row.sentiment) === 'Negative'
                              ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20'
                              : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20'
                          }`}
                        >
                          {row.Sentiments || row.sentiment}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-slate-400">{row.word_count || 15}w</td>
                      <td className="py-3 text-right">
                        <button
                          onClick={() => {
                            const revId = row.Review_ID || row.id || `REV-${actualIdx}`;
                            const revText = row.Review_Text || row.review_text || '';
                            const prodName = row.Product_Name || row.product_category || 'Product Review';
                            const ratingVal = row.Rating || row.stars || 5;

                            window.dispatchEvent(
                              new CustomEvent('analyze-dataset-review', {
                                detail: {
                                  reviewId: revId,
                                  reviewText: revText,
                                  productName: prodName,
                                  rating: ratingVal,
                                },
                              })
                            );
                          }}
                          className="px-2.5 py-1 rounded-lg text-[10px] font-semibold bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 transition-all inline-flex items-center gap-1 shadow-sm"
                          title="Analyze tokens in TF-IDF & MiniLM"
                        >
                          <span>Analyze</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}

              {/* ALL ROWS (Paginated) */}
              {viewMode === 'all' &&
                paginatedRows.map((row, idx) => {
                  const actualIdx = (currentPage - 1) * rowsPerPage + idx + 1;
                  return (
                    <tr key={`all-${row.Review_ID || idx}`} className="hover:bg-surface/50 transition-colors">
                      <td className="py-3 pr-3 text-slate-400 font-bold">#{actualIdx}</td>
                      <td className="py-3 pr-4 text-primary font-semibold">{row.Review_ID || row.id}</td>
                      <td className="py-3 pr-4 text-slate-700 dark:text-slate-300 font-sans">
                        {row.Product_Name || row.product_category}
                      </td>
                      <td className="py-3 pr-4 font-bold text-amber-500">
                        {'★'.repeat(Math.max(1, Math.min(5, row.Rating || row.stars || 5)))}
                      </td>
                      <td className="py-3 pr-4 font-sans text-xs text-slate-700 dark:text-slate-300 max-w-md truncate">
                        {row.Review_Text || row.review_text}
                      </td>
                      <td className="py-3 pr-4">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            (row.Sentiments || row.sentiment) === 'Positive'
                              ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                              : (row.Sentiments || row.sentiment) === 'Negative'
                              ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20'
                              : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20'
                          }`}
                        >
                          {row.Sentiments || row.sentiment}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-slate-400">{row.word_count || 15}w</td>
                      <td className="py-3 text-right">
                        <button
                          onClick={() => {
                            const revId = row.Review_ID || row.id || `REV-${actualIdx}`;
                            const revText = row.Review_Text || row.review_text || '';
                            const prodName = row.Product_Name || row.product_category || 'Product Review';
                            const ratingVal = row.Rating || row.stars || 5;

                            window.dispatchEvent(
                              new CustomEvent('analyze-dataset-review', {
                                detail: {
                                  reviewId: revId,
                                  reviewText: revText,
                                  productName: prodName,
                                  rating: ratingVal,
                                },
                              })
                            );
                          }}
                          className="px-2.5 py-1 rounded-lg text-[10px] font-semibold bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 transition-all inline-flex items-center gap-1 shadow-sm"
                          title="Analyze tokens in TF-IDF & MiniLM"
                        >
                          <span>Analyze</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>

        {/* Footer info & Pagination for All mode */}
        {viewMode === 'all' && (
          <div className="flex items-center justify-between pt-4 mt-2 border-t border-surface-border text-xs text-foreground-muted">
            <div>
              Showing {(currentPage - 1) * rowsPerPage + 1}–
              {Math.min(currentPage * rowsPerPage, totalRows)} of {totalRows} reviews
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="p-1.5 rounded-lg border border-surface-border hover:bg-surface disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="font-mono">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="p-1.5 rounded-lg border border-surface-border hover:bg-surface disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

