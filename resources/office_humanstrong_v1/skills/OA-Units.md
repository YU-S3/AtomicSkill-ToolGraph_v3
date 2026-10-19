# OA-Units

Preserve scales, exchange orientation, and quotations

Read the source unit immediately above the actual numeric column, including nominal dollars, thousands, millions, billions, currency and percent versus percentage points. Convert to the requested unit BEFORE nonlinear transforms; no second scaling at the end. Treasury 99.27 in a table explicitly quoted in 32nds is 99+27/32, not decimal 99.27. Foreign currency per USD versus USD per foreign currency reverses the conversion. Use treasury_transform only with the source convention explicitly established.
