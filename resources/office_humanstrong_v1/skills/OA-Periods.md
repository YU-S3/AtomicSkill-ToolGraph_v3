# OA-Periods

Acquire every required period in a comparable series

Enumerate the actual required calendar months, fiscal months or fiscal/calendar years BEFORE computing a cross-period statistic. Keep a period→value→unit→source ledger, deduplicate real observations, and search ONLY periods still missing. Do not treat Bulletin publication year, adjacent values, a partial table, cumulative-to-date totals or an unconfirmed revised series as equivalent. For trends use correct elapsed intervals, first/last inclusion and observation order. Use period_ledger for local completeness bookkeeping; it cannot supply missing data.
