"""One generation-field boundary shared by E1, runtime drafts and Builder.

These are authoring descriptions, never a substitute for code validators.
"""

OUTPUT_IDENTITY_RULE = (
    "Use input_identity to return the exact typed identity of a declared input. "
    "A newly discovered entity requires effect_witness or another offered, verifiable "
    "output derivation. compatible_with_input states category compatibility only; "
    "it proves neither identity nor an object-location relation."
)

ENTRY_BOUNDARY_RULE = (
    "Entry conditions are requirements that must hold before the first program step. "
    "A condition established by internal preparation belongs at its actual use site "
    "in the IR, with an offered query/guard; it is not an unconditional external "
    "entry requirement. Necessary action failure still fails the program. "
    "Do not remove author conditions or invent evidence to pass admission."
)

CAPABILITY_BOUNDARY_RULES = OUTPUT_IDENTITY_RULE + "\n" + ENTRY_BOUNDARY_RULE
