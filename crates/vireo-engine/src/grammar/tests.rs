use super::*;

fn accepts(schema: &Schema, text: &str) -> bool {
    let g = Grammar::new(schema);
    let mut m = g.matcher();
    m.advance_all(text.as_bytes()) && m.is_complete()
}

fn read_file() -> Schema {
    Schema::Object(vec![
        Property::required("path", Schema::string()),
        Property::optional("max_lines", Schema::Integer),
    ])
}

#[test]
fn objects_in_the_schemas_order_with_optional_keys() {
    let s = read_file();
    assert!(accepts(&s, r#"{"path": "notes.txt"}"#));
    assert!(accepts(&s, r#"{"path": "a", "max_lines": 20}"#));
    assert!(!accepts(&s, r#"{"max_lines": 20, "path": "a"}"#), "order");
    assert!(!accepts(&s, r#"{}"#), "a required key left out");
    assert!(!accepts(&s, r#"{"path":"a"}"#), "one form of spacing");
    assert!(
        !accepts(&s, r#"{"path": "a", "other": 1}"#),
        "an unknown key"
    );
    assert!(!accepts(&s, r#"{"path": "a""#), "not closed");
    let all_optional = Schema::Object(vec![Property::optional("n", Schema::Integer)]);
    assert!(accepts(&all_optional, "{}"));
    assert!(accepts(&all_optional, r#"{"n": 3}"#));
}

#[test]
fn strings_numbers_and_choices() {
    let s = Schema::string();
    for ok in [
        r#""""#,
        r#""plain""#,
        r#""a \"quote\" and \\""#,
        r#""line\nbreak""#,
        r#""é""#,
        "\"caf\u{e9} \u{1f600}\"",
    ] {
        assert!(accepts(&s, ok), "{ok}");
    }
    for bad in [
        "\"raw\nnewline\"",
        r#""bad \x escape""#,
        r#""\u12""#,
        "unquoted",
    ] {
        assert!(!accepts(&s, bad), "{bad}");
    }
    let short = Schema::String { max_bytes: Some(3) };
    assert!(accepts(&short, r#""abc""#));
    assert!(!accepts(&short, r#""abcd""#));

    for ok in ["0", "-7", "120"] {
        assert!(accepts(&Schema::Integer, ok), "{ok}");
    }
    for bad in ["01", "1.5", "-", "+1"] {
        assert!(!accepts(&Schema::Integer, bad), "{bad}");
    }
    for ok in ["0.5", "-2.25e10", "3E-2", "7"] {
        assert!(accepts(&Schema::Number, ok), "{ok}");
    }
    assert!(!accepts(&Schema::Number, "1."));

    let colour = Schema::OneOf(vec!["red".into(), "green".into(), "dark \"green\"".into()]);
    assert!(accepts(&colour, r#""green""#));
    assert!(accepts(&colour, r#""dark \"green\"""#));
    assert!(!accepts(&colour, r#""blue""#));
    assert!(accepts(&Schema::Boolean, "false"));
    assert!(!accepts(&Schema::Boolean, "no"));
}

#[test]
fn arrays_and_nesting() {
    let tags = Schema::Array {
        items: Box::new(Schema::string()),
        max_items: Some(2),
    };
    assert!(accepts(&tags, "[]"));
    assert!(accepts(&tags, r#"["a", "b"]"#));
    assert!(!accepts(&tags, r#"["a", "b", "c"]"#), "too many");
    assert!(!accepts(&tags, r#"["a",]"#));
    let numbers = Schema::Array {
        items: Box::new(Schema::Number),
        max_items: None,
    };
    assert!(
        accepts(&numbers, "[1, 2.5, -3]"),
        "numbers end at the separator"
    );
    let nested = Schema::Object(vec![Property::required(
        "rows",
        Schema::Array {
            items: Box::new(Schema::Object(vec![Property::required(
                "x",
                Schema::Integer,
            )])),
            max_items: None,
        },
    )]);
    assert!(accepts(&nested, r#"{"rows": [{"x": 1}, {"x": 2}]}"#));
}

fn tools() -> Schema {
    Schema::tool_call(&[
        ("read_file".into(), read_file()),
        (
            "list_directory".into(),
            Schema::Object(vec![Property::required("path", Schema::string())]),
        ),
        ("get_current_datetime".into(), Schema::Object(vec![])),
    ])
}

#[test]
fn a_tool_call_with_each_tools_own_arguments() {
    let s = tools();
    assert!(accepts(
        &s,
        r#"{"name": "read_file", "arguments": {"path": "a.txt"}}"#
    ));
    assert!(accepts(
        &s,
        r#"{"name": "get_current_datetime", "arguments": {}}"#
    ));
    assert!(
        !accepts(&s, r#"{"name": "write_file", "arguments": {}}"#),
        "an unknown tool"
    );
    assert!(
        !accepts(
            &s,
            r#"{"name": "list_directory", "arguments": {"path": "a", "max_lines": 3}}"#
        ),
        "another tool's argument"
    );
}

#[test]
fn forced_text_runs_to_the_next_real_choice() {
    let g = Grammar::new(&tools());
    let mut m = g.matcher();
    assert_eq!(m.forced(), br#"{"name": ""#);
    assert!(m.advance_all(br#"{"name": "r"#));
    // One tool starts with r: its name, the arguments' key and the first argument's.
    assert_eq!(m.forced(), br#"ead_file", "arguments": {"path": ""#);
    assert!(m.advance_all(br#"ead_file", "arguments": {"path": "x""#));
    assert!(m.forced().is_empty(), "the optional key or the end");
    assert!(m.advance_all(b"}"));
    assert_eq!(m.forced(), b"}");
    assert!(m.advance_all(b"}"));
    assert!(m.is_complete());
    assert!(m.forced().is_empty());
    assert!(!m.allows(b" "), "nothing after the end");
}

/// The trie's answer is the same as trying every token, wherever the matcher is.
#[test]
fn allowed_tokens_match_trying_every_token() {
    let words = [
        "{",
        "}",
        "{\"",
        "\"",
        "\": \"",
        "\", \"",
        ", ",
        ": ",
        "name",
        "\"name\"",
        "read",
        "_file",
        "list",
        "_directory",
        "path",
        "a",
        "b",
        ".txt",
        "\\",
        "\\n",
        "1",
        "12",
        "-",
        "0",
        "}}",
        "\"}",
        "\"}}",
        " ",
        "caf",
        "\u{e9}",
        "\u{1f600}",
        "arguments",
        "\"arguments\": {",
        "x\"",
    ];
    let mut tokens: Vec<(Vec<u8>, u32)> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_bytes().to_vec(), i as u32))
        .collect();
    // A token of half a character, as byte-level vocabularies have.
    tokens.push((vec![0xC3], tokens.len() as u32));
    tokens.push((vec![0xA9, b'"'], tokens.len() as u32));
    let trie = TokenTrie::from_tokens(tokens.len(), tokens.clone());
    let g = Grammar::new(&tools());
    let text =
        r#"{"name": "read_file", "arguments": {"path": "café x", "max_lines": 12}}"#.as_bytes();
    let mut m = g.matcher();
    for i in 0..=text.len() {
        let allowed = m.allowed(&trie);
        for (bytes, id) in &tokens {
            let want = m.allows(bytes);
            let got = allowed[*id as usize / 64] >> (id % 64) & 1 == 1;
            assert_eq!(
                got,
                want,
                "token {:?} after {:?}",
                String::from_utf8_lossy(bytes),
                String::from_utf8_lossy(&text[..i])
            );
        }
        if i < text.len() {
            assert!(m.advance(text[i]), "at {i}");
        }
    }
    assert!(m.is_complete());
}

/// Qwen 3.5's form: `<parameter=key>` lines, raw values ended by `</parameter>`.
fn qwen_call() -> Schema {
    let end = "\n</parameter>\n";
    let style = RecordStyle {
        open: String::new(),
        key_open: "<parameter=".into(),
        key_close: ">\n".into(),
        separator: String::new(),
        close: "</function>\n".into(),
        json_keys: false,
    };
    let text = || Schema::Text {
        until: end.into(),
        max_bytes: None,
    };
    Schema::Seq(vec![
        Schema::Literal("\n<function=".into()),
        Schema::Branch(vec![(
            "read_file>\n".into(),
            Schema::Record {
                fields: vec![
                    Property::required("path", text()),
                    Property::optional(
                        "max_lines",
                        Schema::Seq(vec![Schema::Integer, Schema::Literal(end.into())]),
                    ),
                ],
                style,
            },
        )]),
    ])
}

#[test]
fn raw_text_values_and_other_punctuation() {
    let q = qwen_call();
    assert!(accepts(
        &q,
        "\n<function=read_file>\n<parameter=path>\nC:\\Users\\me\\notes.txt\n</parameter>\n</function>\n"
    ));
    assert!(
        accepts(
            &q,
            "\n<function=read_file>\n<parameter=path>\nline one\nline two </parameter> \"quoted\"\n</parameter>\n<parameter=max_lines>\n20\n</parameter>\n</function>\n"
        ),
        "a value may hold newlines, quotes and most of its end"
    );
    assert!(
        !accepts(&q, "\n<function=read_file>\n</function>\n"),
        "path is required"
    );

    // Gemma 4's form: call:name{key:<|"|>text<|"|>,n:3}, keys in alphabetical order.
    let quote = "<|\"|>";
    let gemma = Schema::Seq(vec![
        Schema::Literal("call:".into()),
        Schema::Branch(vec![(
            "read_file".into(),
            Schema::Record {
                fields: vec![
                    Property::optional("max_lines", Schema::Integer),
                    Property::required(
                        "path",
                        Schema::Seq(vec![
                            Schema::Literal(quote.into()),
                            Schema::Text {
                                until: quote.into(),
                                max_bytes: None,
                            },
                        ]),
                    ),
                ],
                style: RecordStyle {
                    open: "{".into(),
                    key_open: String::new(),
                    key_close: ":".into(),
                    separator: ",".into(),
                    close: "}".into(),
                    json_keys: false,
                },
            },
        )]),
    ]);
    assert!(accepts(
        &gemma,
        "call:read_file{path:<|\"|>a \"b\" <|c<|\"|>}"
    ));
    assert!(accepts(
        &gemma,
        "call:read_file{max_lines:5,path:<|\"|>x<|\"|>}"
    ));
    assert!(
        !accepts(&gemma, "call:read_file{path:<|\"|>x<|\"|>,max_lines:5}"),
        "order"
    );

    // The trie's tokens, against trying each, through a raw value.
    let words = [
        "\n",
        "\n<",
        "<",
        "</",
        "parameter",
        ">\n",
        "</parameter>",
        "\n</parameter>\n",
        "<parameter=",
        "path",
        "C:\\",
        "Users",
        "\\me",
        "\\notes",
        ".txt",
        "line",
        " two",
        "function",
        "</function>\n",
        "\t",
        "\"",
        "x\n",
        "\u{e9}",
        "read_file",
        ">",
    ];
    let mut tokens: Vec<(Vec<u8>, u32)> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_bytes().to_vec(), i as u32))
        .collect();
    tokens.push((vec![0xC3], tokens.len() as u32));
    let trie = TokenTrie::from_tokens(tokens.len(), tokens.clone());
    let g = Grammar::new(&q);
    let text = "\n<function=read_file>\n<parameter=path>\nC:\\Users\\me\\caf\u{e9}\n</para\tx\n</parameter>\n</function>\n".as_bytes();
    let mut m = g.matcher();
    for i in 0..=text.len() {
        let allowed = m.allowed(&trie);
        for (bytes, id) in &tokens {
            let got = allowed[*id as usize / 64] >> (id % 64) & 1 == 1;
            assert_eq!(
                got,
                m.allows(bytes),
                "token {:?} after {:?}",
                String::from_utf8_lossy(bytes),
                String::from_utf8_lossy(&text[..i])
            );
        }
        if i < text.len() {
            assert!(m.advance(text[i]), "at {i}");
        }
    }
    assert!(m.is_complete());
}
