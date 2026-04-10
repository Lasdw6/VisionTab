"""
Tree-sitter language configurations for AST-aware FIM masking.
Each language defines which AST node types are valid masking targets,
aligned with real-world editing patterns per [AST-FIM] paper.
"""

LANGUAGE_CONFIGS = {
    "python": {
        "module": "tree_sitter_python",
        "extensions": [".py"],
        "maskable_node_types": [
            "block",
            "expression_statement",
            "return_statement",
            "assignment",
            "augmented_assignment",
            "call",
            "if_statement",
            "for_statement",
            "while_statement",
            "with_statement",
            "try_statement",
            "function_definition",
            "class_definition",
            "list_comprehension",
            "dictionary_comprehension",
            "conditional_expression",
            "lambda",
        ],
    },
    "javascript": {
        "module": "tree_sitter_javascript",
        "extensions": [".js", ".jsx", ".mjs"],
        "maskable_node_types": [
            "statement_block",
            "expression_statement",
            "return_statement",
            "lexical_declaration",
            "variable_declaration",
            "assignment_expression",
            "call_expression",
            "if_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "switch_statement",
            "try_statement",
            "function_declaration",
            "arrow_function",
            "class_declaration",
            "template_string",
            "ternary_expression",
            "object",
            "array",
        ],
    },
    "typescript": {
        "module": "tree_sitter_typescript",
        "language_attr": "language_typescript",
        "extensions": [".ts", ".tsx"],
        "maskable_node_types": [
            "statement_block",
            "expression_statement",
            "return_statement",
            "lexical_declaration",
            "variable_declaration",
            "assignment_expression",
            "call_expression",
            "if_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "switch_statement",
            "try_statement",
            "function_declaration",
            "arrow_function",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "template_string",
            "ternary_expression",
            "type_annotation",
            "object",
            "array",
        ],
    },
    "rust": {
        "module": "tree_sitter_rust",
        "extensions": [".rs"],
        "maskable_node_types": [
            "block",
            "expression_statement",
            "let_declaration",
            "return_expression",
            "call_expression",
            "if_expression",
            "match_expression",
            "for_expression",
            "while_expression",
            "loop_expression",
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "closure_expression",
            "macro_invocation",
            "tuple_expression",
            "array_expression",
        ],
    },
    "go": {
        "module": "tree_sitter_go",
        "extensions": [".go"],
        "maskable_node_types": [
            "block",
            "expression_statement",
            "return_statement",
            "short_var_declaration",
            "assignment_statement",
            "call_expression",
            "if_statement",
            "for_statement",
            "switch_statement",
            "select_statement",
            "function_declaration",
            "method_declaration",
            "go_statement",
            "defer_statement",
            "composite_literal",
        ],
    },
    "java": {
        "module": "tree_sitter_java",
        "extensions": [".java"],
        "maskable_node_types": [
            "block",
            "expression_statement",
            "return_statement",
            "local_variable_declaration",
            "assignment_expression",
            "method_invocation",
            "if_statement",
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "switch_expression",
            "try_statement",
            "method_declaration",
            "class_declaration",
            "lambda_expression",
            "object_creation_expression",
        ],
    },
}


def get_supported_languages():
    return list(LANGUAGE_CONFIGS.keys())


def get_extensions_for_language(lang: str) -> list[str]:
    return LANGUAGE_CONFIGS.get(lang, {}).get("extensions", [])


def get_all_extensions() -> dict[str, str]:
    """Returns a mapping of file extension -> language name."""
    ext_map = {}
    for lang, config in LANGUAGE_CONFIGS.items():
        for ext in config["extensions"]:
            ext_map[ext] = lang
    return ext_map
