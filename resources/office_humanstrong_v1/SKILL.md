# OfficeQA HumanStrong Programmatic Bank v1

This is an authored reference Bank. The Runtime reads Bank guidance; not this file by default.

## OA-Target

Build a task target with exact measure, entity, observation time, reporting vintage, calendar/fiscal basis, required periods, input unit, requested operation and final answer form. A Treasury Bulletin issue date is not necessarily the date of the observation. Treat the question as an instruction to obtain actual source values, not to infer them from generic financial knowledge. Once every operand is available, finish rather than restarting retrieval.

## OA-Locate

Search a bounded candidate file scope using the exact table family and measure/period; use glob→grep_page→read_context with path and offset from the SAME real hit. Read nearby title/header/unit. If one anchor fails, change one term or issue-period dimension; do not repeat the same global search. A search hit or even search_exhausted=true means nothing about whether the requested observations are complete. Stop searching when actual required values have been acquired.

## OA-Table

Treat a Treasury table as (table title, section, parent headers, exact column, row, observation period, scale, footnote), not a bag of nearby numbers. Find the exact requested row then follow its header hierarchy across column boundaries and continued pages. An amount is not a percent or a subtotal; avoid OCR columns shifted into neighboring tables. Bind each value to the acquired source window; if row/header remains ambiguous read a smaller window rather than fabricating a cell.

## OA-Periods

Enumerate the actual required calendar months, fiscal months or fiscal/calendar years BEFORE computing a cross-period statistic. Keep a period→value→unit→source ledger, deduplicate real observations, and search ONLY periods still missing. Do not treat Bulletin publication year, adjacent values, a partial table, cumulative-to-date totals or an unconfirmed revised series as equivalent. For trends use correct elapsed intervals, first/last inclusion and observation order. Use period_ledger for local completeness bookkeeping; it cannot supply missing data.

## OA-Treasury

Distinguish receipts/outlays by agency versus budget function; offered, tendered, accepted, exchanged, refunded and new cash; amount issued, amount outstanding, retired and maturity bucket; marketable versus nonmarketable and government-account holdings versus debt held by the public. For bill/note/bond tables use issue date AND maturity date, and the exact security class. For effective interest rates use a rate table or an explicitly requested interest/outstanding calculation, not net-interest outlays. Never substitute a neighboring field merely because it shares a plausible amount.

## OA-Units

Read the source unit immediately above the actual numeric column, including nominal dollars, thousands, millions, billions, currency and percent versus percentage points. Convert to the requested unit BEFORE nonlinear transforms; no second scaling at the end. Treasury 99.27 in a table explicitly quoted in 32nds is 99+27/32, not decimal 99.27. Foreign currency per USD versus USD per foreign currency reverses the conversion. Use treasury_transform only with the source convention explicitly established.

## OA-Stats

Before computation identify operation, numerator/denominator or from/to sign, full ordered series, elapsed periods, and rounding. CAGR=(end/start)^(1/n)-1, log growth=ln(end/start)/n, geometric growth of rates uses product(1+r), Pearson/OLS uses aligned pairs, sample stdev uses n-1 and population uses n. Confirm input dates/unit before any local Program. For series_metrics use named parameters; the Program proves arithmetic on supplied numbers only, not that they were the correct source operands.

## OA-Fiscal

Historic U.S. federal fiscal years generally ended June 30 before the transition in 1976; FY1977 onward ends September 30, with a special July–September 1976 transition quarter. Never mechanically apply the modern date rule to older Treasury Bulletins. Bare month labels may inherit the year from a preceding header; derive it from the actual table, not file name. Separate fiscal month, calendar month, as-of date and issue month.

## OA-Advanced

For ARIMA or similar forecasting, identify model order, differencing, horizon, input sequence and fitting method; do not call simple OLS a substitute for ARIMA. For Box-Cox confirm positive values, lambda and scale first; lambda=0 uses ln(x), otherwise (x^lambda−1)/lambda. For quantitative rankings compute the exact within-row gap/statistic before ordering. If the operation is not available in current Programs, use permitted temporary Python only if the algorithm and full required data are actually available.

## OA-Commit

When the required facts and computation are established, immediately call the native runtime_step finish action and provide only the requested number, date or list. Do not add prose, labels, currency/unit symbols, or a percent glyph to a numeric answer unless the exact output template explicitly requires it; preserve percent SCALE separately from the percent SYMBOL. Keep the source-derived sign, rounding and requested list ordering. A final numeric candidate may be in local Program/Python output: reference its actual scalar field and check target/period/unit. Do not search again for an already established target.

## OA-Fallback

If the required period or column is absent, identify the exact missing measure and pursue that narrow gap while tool budget remains. Do not substitute remembered Treasury figures or invented continuation pages. An acquired-window Program returning ok is not whole-task success; a local arithmetic Program returning ok is not evidence of correct input semantics. Near the budget limit, retain the strongest actual evidence and use the native final plain-text protocol; do not request tools during finish_only or invent successful verification.
