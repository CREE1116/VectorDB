//! Structural code & document parser.
//! Extracts semantic chunks (functions, classes, sections) and builds knowledge graph nodes/edges.

use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

use vectordb_core::{Chunk, Edge, EdgeKind, Node, NodeKind};

#[derive(Debug, Clone)]
pub struct ParsedFile {
    pub file_path: String,
    pub chunks: Vec<Chunk>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Default)]
pub struct CodeParser;

impl CodeParser {
    pub fn new() -> Self {
        Self
    }

    pub fn is_supported(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        !crate::universal::UniversalFileAdapter::is_excluded_extension(&ext)
    }

    /// Parse a single file into semantic chunks and graph elements.
    /// Only readable UTF-8 text/code and PDFs with extracted text are indexed.
    pub fn parse_file(&self, path: &Path, rel_path: &str) -> anyhow::Result<ParsedFile> {
        anyhow::ensure!(self.is_supported(path), "non-text file is not supported");
        let bytes = fs::read(path)?;
        let format = crate::universal::UniversalFileAdapter::detect_format(&bytes, path);

        let parsed: anyhow::Result<ParsedFile> = match format {
            crate::universal::FileCategory::Pdf => {
                let pf = crate::universal::UniversalFileAdapter::parse_pdf(&bytes, rel_path);
                anyhow::ensure!(!pf.chunks.is_empty(), "PDF has no extractable text");
                Ok(pf)
            }
            crate::universal::FileCategory::TextOrCode => {
                let content = std::str::from_utf8(&bytes)?;
                anyhow::ensure!(!content.trim().is_empty(), "empty text file");
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                let file_node_id = format!("file:{}", rel_path);
                let file_node = Node {
                    id: file_node_id.clone(),
                    label: Path::new(rel_path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                    kind: NodeKind::File,
                    file_path: rel_path.to_string(),
                    start_line: 1,
                    end_line: content.lines().count().max(1),
                    signature: None,
                    docstring: None,
                    chunk_id: None,
                    vector_id: None,
                    metadata: HashMap::new(),
                };

                let mut chunks = Vec::new();
                let mut nodes = vec![file_node];
                let mut edges = Vec::new();

                match ext.as_str() {
                    "rs" => self.parse_rust(
                        rel_path,
                        content,
                        &file_node_id,
                        &mut chunks,
                        &mut nodes,
                        &mut edges,
                    ),
                    "py" => self.parse_python(
                        rel_path,
                        content,
                        &file_node_id,
                        &mut chunks,
                        &mut nodes,
                        &mut edges,
                    ),
                    "js" | "ts" | "jsx" | "tsx" => self.parse_javascript(
                        rel_path,
                        content,
                        &file_node_id,
                        &mut chunks,
                        &mut nodes,
                        &mut edges,
                    ),
                    "md" => self.parse_markdown(
                        rel_path,
                        content,
                        &file_node_id,
                        &mut chunks,
                        &mut nodes,
                        &mut edges,
                    ),
                    _ => self.parse_generic(
                        rel_path,
                        content,
                        &file_node_id,
                        &mut chunks,
                        &mut nodes,
                        &mut edges,
                    ),
                }

                Ok(ParsedFile {
                    file_path: rel_path.to_string(),
                    chunks,
                    nodes,
                    edges,
                })
            }
            _ => anyhow::bail!("non-text file is not supported"),
        };
        let mut parsed = parsed?;
        Self::normalize_chunk_edges(&mut parsed);
        Ok(parsed)
    }

    fn normalize_chunk_edges(file: &mut ParsedFile) {
        let mut chunk_nodes = HashMap::new();
        for chunk in &mut file.chunks {
            if chunk.node_id.is_none() {
                chunk.node_id = Some(chunk.id.clone());
                file.nodes.push(Node {
                    id: chunk.id.clone(),
                    label: chunk.id.clone(),
                    kind: NodeKind::Chunk,
                    file_path: chunk.file_path.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    signature: None,
                    docstring: None,
                    chunk_id: Some(chunk.id.clone()),
                    vector_id: None,
                    metadata: HashMap::new(),
                });
            }
            chunk_nodes.insert(chunk.id.clone(), chunk.node_id.as_ref().unwrap().clone());
        }
        for edge in &mut file.edges {
            if let Some(node_id) = chunk_nodes.get(&edge.source) {
                edge.source = node_id.clone();
            }
            if let Some(node_id) = chunk_nodes.get(&edge.target) {
                edge.target = node_id.clone();
            }
        }
    }

    fn find_closing_brace(lines: &[&str], start_idx: usize) -> usize {
        let mut brace_depth = 0;
        let mut found_brace = false;

        for (k, line) in lines.iter().enumerate().skip(start_idx) {
            for ch in line.chars() {
                if ch == '{' {
                    brace_depth += 1;
                    found_brace = true;
                } else if ch == '}' {
                    brace_depth -= 1;
                }
            }
            if found_brace && brace_depth <= 0 {
                return k + 1;
            }
        }
        lines.len()
    }

    // --------------------------------------------------------------------------------
    // Rust Parser
    // --------------------------------------------------------------------------------
    fn parse_rust(
        &self,
        rel_path: &str,
        content: &str,
        file_node_id: &str,
        chunks: &mut Vec<Chunk>,
        nodes: &mut Vec<Node>,
        edges: &mut Vec<Edge>,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        let fn_regex =
            Regex::new(r"^(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)").unwrap();
        let struct_regex =
            Regex::new(r"^(?:pub(?:\s*\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([a-zA-Z0-9_]+)")
                .unwrap();
        let impl_trait_regex = Regex::new(
            r"^impl(?:\s*<[^>]*>)?\s+([a-zA-Z0-9_:]+)(?:\s*<[^>]*>)?\s+for\s+([a-zA-Z0-9_:]+)",
        )
        .unwrap();
        let impl_struct_regex = Regex::new(r"^impl(?:\s*<[^>]*>)?\s+([a-zA-Z0-9_:]+)").unwrap();

        let mut i = 0;
        let mut last_chunk_id: Option<String> = None;

        while i < lines.len() {
            let mut doc_lines = Vec::new();
            let mut symbol_line_idx = i;
            while symbol_line_idx < lines.len() && lines[symbol_line_idx].trim().starts_with("///")
            {
                doc_lines.push(
                    lines[symbol_line_idx]
                        .trim()
                        .trim_start_matches("///")
                        .trim(),
                );
                symbol_line_idx += 1;
            }

            if symbol_line_idx >= lines.len() {
                break;
            }

            let test_line = lines[symbol_line_idx].trim();

            // 1. Check impl blocks (impl Trait for Struct)
            if let Some(caps) = impl_trait_regex.captures(test_line) {
                let trait_name = caps[1].to_string();
                let struct_name = caps[2].to_string();
                let impl_end = Self::find_closing_brace(&lines, symbol_line_idx);

                let impl_node_id = format!(
                    "symbol:{}:impl_{}_for_{}",
                    rel_path, trait_name, struct_name
                );
                let impl_node = Node {
                    id: impl_node_id.clone(),
                    label: format!("impl {} for {}", trait_name, struct_name),
                    kind: NodeKind::Trait,
                    file_path: rel_path.to_string(),
                    start_line: symbol_line_idx + 1,
                    end_line: impl_end,
                    signature: Some(format!("impl {} for {}", trait_name, struct_name)),
                    docstring: if doc_lines.is_empty() {
                        None
                    } else {
                        Some(doc_lines.join(" "))
                    },
                    chunk_id: None,
                    vector_id: None,
                    metadata: HashMap::new(),
                };
                nodes.push(impl_node);

                edges.push(Edge {
                    source: file_node_id.to_string(),
                    target: impl_node_id.clone(),
                    kind: EdgeKind::Contains,
                    weight: 1.0,
                });

                // Structural Implements Edge: Struct -> Trait
                edges.push(Edge {
                    source: format!("symbol:{}:{}", rel_path, struct_name),
                    target: format!("symbol:{}", trait_name),
                    kind: EdgeKind::Implements,
                    weight: 1.0,
                });

                // Parse methods inside this impl block
                let mut method_idx = symbol_line_idx + 1;
                let mut methods_found = 0;
                while method_idx < impl_end {
                    let m_line = lines[method_idx].trim();
                    if let Some(m_caps) = fn_regex.captures(m_line) {
                        let m_name = m_caps[1].to_string();
                        let m_end = Self::find_closing_brace(&lines, method_idx);

                        let m_lines = &lines[method_idx..m_end];
                        let m_content = m_lines.join("\n");
                        let m_chunk_id = format!("chunk:{}:{}:{}", rel_path, method_idx + 1, m_end);
                        let m_node_id = format!("symbol:{}:{}::{}", rel_path, struct_name, m_name);

                        let m_node = Node {
                            id: m_node_id.clone(),
                            label: format!("{}::{}", struct_name, m_name),
                            kind: NodeKind::Method,
                            file_path: rel_path.to_string(),
                            start_line: method_idx + 1,
                            end_line: m_end,
                            signature: Some(format!(
                                "impl {} for {} -> fn {}",
                                trait_name, struct_name, m_name
                            )),
                            docstring: None,
                            chunk_id: Some(m_chunk_id.clone()),
                            vector_id: None,
                            metadata: HashMap::new(),
                        };

                        let m_chunk = Chunk {
                            id: m_chunk_id.clone(),
                            file_path: rel_path.to_string(),
                            start_line: method_idx + 1,
                            end_line: m_end,
                            tokens_approx: m_content.split_whitespace().count(),
                            content: m_content,
                            summary: None,
                            node_id: Some(m_node_id.clone()),
                            vector_id: 0,
                        };

                        edges.push(Edge {
                            source: impl_node_id.clone(),
                            target: m_node_id.clone(),
                            kind: EdgeKind::Defines,
                            weight: 1.0,
                        });
                        edges.push(Edge {
                            source: file_node_id.to_string(),
                            target: m_node_id.clone(),
                            kind: EdgeKind::Contains,
                            weight: 1.0,
                        });
                        edges.push(Edge {
                            source: m_node_id.clone(),
                            target: format!("symbol:{}::{}", trait_name, m_name),
                            kind: EdgeKind::Implements,
                            weight: 0.9,
                        });

                        if let Some(prev) = last_chunk_id {
                            edges.push(Edge {
                                source: prev,
                                target: m_chunk_id.clone(),
                                kind: EdgeKind::NextChunk,
                                weight: 0.5,
                            });
                        }
                        last_chunk_id = Some(m_chunk_id.clone());

                        nodes.push(m_node);
                        chunks.push(m_chunk);
                        methods_found += 1;
                        method_idx = m_end;
                        continue;
                    }
                    method_idx += 1;
                }

                if methods_found == 0 {
                    let chunk_lines = &lines[i..impl_end];
                    let chunk_content = chunk_lines.join("\n");
                    let chunk_id = format!("chunk:{}:{}:{}", rel_path, i + 1, impl_end);
                    let chunk = Chunk {
                        id: chunk_id.clone(),
                        file_path: rel_path.to_string(),
                        start_line: i + 1,
                        end_line: impl_end,
                        tokens_approx: chunk_content.split_whitespace().count(),
                        content: chunk_content,
                        summary: None,
                        node_id: Some(impl_node_id.clone()),
                        vector_id: 0,
                    };
                    chunks.push(chunk);
                }

                i = impl_end;
                continue;
            } else if let Some(caps) = impl_struct_regex.captures(test_line) {
                let struct_name = caps[1].to_string();
                let impl_end = Self::find_closing_brace(&lines, symbol_line_idx);

                // Parse methods inside this impl block
                let mut method_idx = symbol_line_idx + 1;
                let mut methods_found = 0;
                while method_idx < impl_end {
                    let m_line = lines[method_idx].trim();
                    if let Some(m_caps) = fn_regex.captures(m_line) {
                        let m_name = m_caps[1].to_string();
                        let m_end = Self::find_closing_brace(&lines, method_idx);

                        let m_lines = &lines[method_idx..m_end];
                        let m_content = m_lines.join("\n");
                        let m_chunk_id = format!("chunk:{}:{}:{}", rel_path, method_idx + 1, m_end);
                        let m_node_id = format!("symbol:{}:{}::{}", rel_path, struct_name, m_name);

                        let m_node = Node {
                            id: m_node_id.clone(),
                            label: format!("{}::{}", struct_name, m_name),
                            kind: NodeKind::Method,
                            file_path: rel_path.to_string(),
                            start_line: method_idx + 1,
                            end_line: m_end,
                            signature: Some(format!("fn {}::{}", struct_name, m_name)),
                            docstring: None,
                            chunk_id: Some(m_chunk_id.clone()),
                            vector_id: None,
                            metadata: HashMap::new(),
                        };

                        let m_chunk = Chunk {
                            id: m_chunk_id.clone(),
                            file_path: rel_path.to_string(),
                            start_line: method_idx + 1,
                            end_line: m_end,
                            tokens_approx: m_content.split_whitespace().count(),
                            content: m_content,
                            summary: None,
                            node_id: Some(m_node_id.clone()),
                            vector_id: 0,
                        };

                        edges.push(Edge {
                            source: format!("symbol:{}:{}", rel_path, struct_name),
                            target: m_node_id.clone(),
                            kind: EdgeKind::Defines,
                            weight: 1.0,
                        });
                        edges.push(Edge {
                            source: file_node_id.to_string(),
                            target: m_node_id.clone(),
                            kind: EdgeKind::Contains,
                            weight: 1.0,
                        });

                        if let Some(prev) = last_chunk_id {
                            edges.push(Edge {
                                source: prev,
                                target: m_chunk_id.clone(),
                                kind: EdgeKind::NextChunk,
                                weight: 0.5,
                            });
                        }
                        last_chunk_id = Some(m_chunk_id.clone());

                        nodes.push(m_node);
                        chunks.push(m_chunk);
                        methods_found += 1;
                        method_idx = m_end;
                        continue;
                    }
                    method_idx += 1;
                }

                if methods_found == 0 {
                    let chunk_lines = &lines[i..impl_end];
                    let chunk_content = chunk_lines.join("\n");
                    let chunk_id = format!("chunk:{}:{}:{}", rel_path, i + 1, impl_end);
                    let node_id = format!("symbol:{}:impl_{}", rel_path, struct_name);
                    let node = Node {
                        id: node_id.clone(),
                        label: format!("impl {}", struct_name),
                        kind: NodeKind::Module,
                        file_path: rel_path.to_string(),
                        start_line: i + 1,
                        end_line: impl_end,
                        signature: Some(format!("impl {}", struct_name)),
                        docstring: None,
                        chunk_id: Some(chunk_id.clone()),
                        vector_id: None,
                        metadata: HashMap::new(),
                    };
                    let chunk = Chunk {
                        id: chunk_id.clone(),
                        file_path: rel_path.to_string(),
                        start_line: i + 1,
                        end_line: impl_end,
                        tokens_approx: chunk_content.split_whitespace().count(),
                        content: chunk_content,
                        summary: None,
                        node_id: Some(node_id.clone()),
                        vector_id: 0,
                    };
                    nodes.push(node);
                    chunks.push(chunk);
                }

                i = impl_end;
                continue;
            }

            // 2. Standalone functions, structs, traits, enums
            let mut detected_symbol: Option<(NodeKind, String, String)> = None;

            if let Some(caps) = fn_regex.captures(test_line) {
                let name = caps[1].to_string();
                detected_symbol = Some((NodeKind::Function, name.clone(), format!("fn {}", name)));
            } else if let Some(caps) = struct_regex.captures(test_line) {
                let name = caps[1].to_string();
                let kind = if test_line.contains("trait ") {
                    NodeKind::Trait
                } else if test_line.contains("enum ") {
                    NodeKind::Class
                } else {
                    NodeKind::Struct
                };
                detected_symbol = Some((kind, name.clone(), format!("type {}", name)));
            }

            if let Some((kind, name, sig)) = detected_symbol {
                let start_line = i + 1;
                let end_line = Self::find_closing_brace(&lines, symbol_line_idx);

                let chunk_lines = &lines[i..end_line];
                let chunk_content = chunk_lines.join("\n");
                let chunk_id = format!("chunk:{}:{}:{}", rel_path, start_line, end_line);
                let node_id = format!("symbol:{}:{}", rel_path, name);

                let docstring = if doc_lines.is_empty() {
                    None
                } else {
                    Some(doc_lines.join(" "))
                };

                let node = Node {
                    id: node_id.clone(),
                    label: name,
                    kind,
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    signature: Some(sig),
                    docstring,
                    chunk_id: Some(chunk_id.clone()),
                    vector_id: None,
                    metadata: HashMap::new(),
                };

                let chunk = Chunk {
                    id: chunk_id.clone(),
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    tokens_approx: chunk_content.split_whitespace().count(),
                    content: chunk_content,
                    summary: None,
                    node_id: Some(node_id.clone()),
                    vector_id: 0,
                };

                edges.push(Edge {
                    source: file_node_id.to_string(),
                    target: node_id.clone(),
                    kind: EdgeKind::Contains,
                    weight: 1.0,
                });

                if let Some(prev) = last_chunk_id {
                    edges.push(Edge {
                        source: prev,
                        target: chunk_id.clone(),
                        kind: EdgeKind::NextChunk,
                        weight: 0.5,
                    });
                }
                last_chunk_id = Some(chunk_id.clone());

                nodes.push(node);
                chunks.push(chunk);

                i = end_line;
                continue;
            }

            i += 1;
        }

        // If no symbols found, fallback to block chunking
        if chunks.is_empty() {
            self.parse_generic(rel_path, content, file_node_id, chunks, nodes, edges);
        }
    }

    // --------------------------------------------------------------------------------
    // Python Parser
    // --------------------------------------------------------------------------------
    fn parse_python(
        &self,
        rel_path: &str,
        content: &str,
        file_node_id: &str,
        chunks: &mut Vec<Chunk>,
        nodes: &mut Vec<Node>,
        edges: &mut Vec<Edge>,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        let def_regex =
            Regex::new(r"^(?:\s*)(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)").unwrap();
        let class_regex = Regex::new(r"^(?:\s*)class\s+([a-zA-Z0-9_]+)(?:\s*\((.*?)\))?").unwrap();

        let mut i = 0;
        let mut last_chunk_id: Option<String> = None;

        while i < lines.len() {
            let line = lines[i];
            let indent = line.len() - line.trim_start().len();

            let mut symbol_info: Option<(NodeKind, String, String, Vec<String>)> = None;

            if let Some(caps) = def_regex.captures(line) {
                let name = caps[1].to_string();
                let kind = if indent > 0 {
                    NodeKind::Method
                } else {
                    NodeKind::Function
                };
                symbol_info = Some((
                    kind,
                    name.clone(),
                    format!("def {}({})", name, &caps[2]),
                    Vec::new(),
                ));
            } else if let Some(caps) = class_regex.captures(line) {
                let name = caps[1].to_string();
                let mut bases = Vec::new();
                if let Some(base_match) = caps.get(2) {
                    for b in base_match.as_str().split(',') {
                        let trimmed = b.trim();
                        if !trimmed.is_empty() && trimmed != "object" {
                            bases.push(trimmed.to_string());
                        }
                    }
                }
                symbol_info = Some((
                    NodeKind::Class,
                    name.clone(),
                    format!("class {}", name),
                    bases,
                ));
            }

            if let Some((kind, name, sig, bases)) = symbol_info {
                let start_line = i + 1;
                let mut end_line = start_line;

                // Python block continues until a non-empty line has indentation <= current indent
                let mut j = i + 1;
                while j < lines.len() {
                    let next_line = lines[j];
                    if !next_line.trim().is_empty() {
                        let next_indent = next_line.len() - next_line.trim_start().len();
                        if next_indent <= indent {
                            break;
                        }
                    }
                    end_line = j + 1;
                    j += 1;
                }

                let chunk_lines = &lines[i..end_line];
                let chunk_content = chunk_lines.join("\n");
                let chunk_id = format!("chunk:{}:{}:{}", rel_path, start_line, end_line);
                let node_id = format!("symbol:{}:{}", rel_path, name);

                let node = Node {
                    id: node_id.clone(),
                    label: name,
                    kind,
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    signature: Some(sig),
                    docstring: None,
                    chunk_id: Some(chunk_id.clone()),
                    vector_id: None,
                    metadata: HashMap::new(),
                };

                let chunk = Chunk {
                    id: chunk_id.clone(),
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    tokens_approx: chunk_content.split_whitespace().count(),
                    content: chunk_content,
                    summary: None,
                    node_id: Some(node_id.clone()),
                    vector_id: 0,
                };

                edges.push(Edge {
                    source: file_node_id.to_string(),
                    target: node_id.clone(),
                    kind: EdgeKind::Contains,
                    weight: 1.0,
                });

                for base in &bases {
                    edges.push(Edge {
                        source: node_id.clone(),
                        target: format!("symbol:{}", base),
                        kind: EdgeKind::Implements,
                        weight: 1.0,
                    });
                }

                if let Some(prev) = last_chunk_id {
                    edges.push(Edge {
                        source: prev,
                        target: chunk_id.clone(),
                        kind: EdgeKind::NextChunk,
                        weight: 0.5,
                    });
                }
                last_chunk_id = Some(chunk_id.clone());

                nodes.push(node);
                chunks.push(chunk);

                i = end_line;
                continue;
            }

            i += 1;
        }

        if chunks.is_empty() {
            self.parse_generic(rel_path, content, file_node_id, chunks, nodes, edges);
        }
    }

    // --------------------------------------------------------------------------------
    // JavaScript / TypeScript Parser
    // --------------------------------------------------------------------------------
    fn parse_javascript(
        &self,
        rel_path: &str,
        content: &str,
        file_node_id: &str,
        chunks: &mut Vec<Chunk>,
        nodes: &mut Vec<Node>,
        edges: &mut Vec<Edge>,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        let fn_regex =
            Regex::new(r"^(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)").unwrap();
        let arrow_regex =
            Regex::new(r"^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\(")
                .unwrap();
        let class_regex = Regex::new(r"^(?:export\s+)?(?:default\s+)?class\s+([a-zA-Z0-9_]+)(?:\s+extends\s+([a-zA-Z0-9_]+))?(?:\s+implements\s+([a-zA-Z0-9_,\s]+))?").unwrap();
        let iface_regex = Regex::new(
            r"^(?:export\s+)?interface\s+([a-zA-Z0-9_]+)(?:\s+extends\s+([a-zA-Z0-9_,\s]+))?",
        )
        .unwrap();

        let mut i = 0;
        let mut last_chunk_id: Option<String> = None;

        while i < lines.len() {
            let line = lines[i].trim();
            let mut detected: Option<(NodeKind, String, String, Vec<String>)> = None;

            if let Some(caps) = fn_regex.captures(line) {
                let name = caps[1].to_string();
                detected = Some((
                    NodeKind::Function,
                    name.clone(),
                    format!("function {}", name),
                    Vec::new(),
                ));
            } else if let Some(caps) = arrow_regex.captures(line) {
                let name = caps[1].to_string();
                detected = Some((
                    NodeKind::Function,
                    name.clone(),
                    format!("const {} = () =>", name),
                    Vec::new(),
                ));
            } else if let Some(caps) = class_regex.captures(line) {
                let name = caps[1].to_string();
                let mut relations = Vec::new();
                if let Some(ext) = caps.get(2) {
                    relations.push(ext.as_str().trim().to_string());
                }
                if let Some(impls) = caps.get(3) {
                    for im in impls.as_str().split(',') {
                        let t = im.trim();
                        if !t.is_empty() {
                            relations.push(t.to_string());
                        }
                    }
                }
                detected = Some((
                    NodeKind::Class,
                    name.clone(),
                    format!("class {}", name),
                    relations,
                ));
            } else if let Some(caps) = iface_regex.captures(line) {
                let name = caps[1].to_string();
                let mut relations = Vec::new();
                if let Some(exts) = caps.get(2) {
                    for ext in exts.as_str().split(',') {
                        let t = ext.trim();
                        if !t.is_empty() {
                            relations.push(t.to_string());
                        }
                    }
                }
                detected = Some((
                    NodeKind::Interface,
                    name.clone(),
                    format!("interface {}", name),
                    relations,
                ));
            }

            if let Some((kind, name, sig, relations)) = detected {
                let start_line = i + 1;
                let end_line = Self::find_closing_brace(&lines, i);

                let chunk_lines = &lines[i..end_line];
                let chunk_content = chunk_lines.join("\n");
                let chunk_id = format!("chunk:{}:{}:{}", rel_path, start_line, end_line);
                let node_id = format!("symbol:{}:{}", rel_path, name);

                let node = Node {
                    id: node_id.clone(),
                    label: name,
                    kind,
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    signature: Some(sig),
                    docstring: None,
                    chunk_id: Some(chunk_id.clone()),
                    vector_id: None,
                    metadata: HashMap::new(),
                };

                let chunk = Chunk {
                    id: chunk_id.clone(),
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    tokens_approx: chunk_content.split_whitespace().count(),
                    content: chunk_content,
                    summary: None,
                    node_id: Some(node_id.clone()),
                    vector_id: 0,
                };

                edges.push(Edge {
                    source: file_node_id.to_string(),
                    target: node_id.clone(),
                    kind: EdgeKind::Contains,
                    weight: 1.0,
                });

                for rel in &relations {
                    edges.push(Edge {
                        source: node_id.clone(),
                        target: format!("symbol:{}", rel),
                        kind: EdgeKind::Implements,
                        weight: 1.0,
                    });
                }

                if let Some(prev) = last_chunk_id {
                    edges.push(Edge {
                        source: prev,
                        target: chunk_id.clone(),
                        kind: EdgeKind::NextChunk,
                        weight: 0.5,
                    });
                }
                last_chunk_id = Some(chunk_id.clone());

                nodes.push(node);
                chunks.push(chunk);

                i = end_line;
                continue;
            }

            i += 1;
        }

        if chunks.is_empty() {
            self.parse_generic(rel_path, content, file_node_id, chunks, nodes, edges);
        }
    }

    // --------------------------------------------------------------------------------
    // Markdown Parser
    // --------------------------------------------------------------------------------
    fn parse_markdown(
        &self,
        rel_path: &str,
        content: &str,
        file_node_id: &str,
        chunks: &mut Vec<Chunk>,
        nodes: &mut Vec<Node>,
        edges: &mut Vec<Edge>,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        let header_regex = Regex::new(r"^(#{1,6})\s+(.*)").unwrap();

        let mut current_section_start = 0;
        let mut current_title = "Introduction".to_string();
        let mut last_chunk_id: Option<String> = None;

        for i in 0..lines.len() {
            if let Some(caps) = header_regex.captures(lines[i]) {
                if i > current_section_start {
                    let section_lines = &lines[current_section_start..i];
                    let section_content = section_lines.join("\n").trim().to_string();
                    if !section_content.is_empty() {
                        let start_line = current_section_start + 1;
                        let end_line = i;
                        let chunk_id = format!("chunk:{}:{}:{}", rel_path, start_line, end_line);
                        let node_id = format!("doc:{}:{}", rel_path, current_title);

                        let node = Node {
                            id: node_id.clone(),
                            label: current_title.clone(),
                            kind: NodeKind::DocSection,
                            file_path: rel_path.to_string(),
                            start_line,
                            end_line,
                            signature: None,
                            docstring: None,
                            chunk_id: Some(chunk_id.clone()),
                            vector_id: None,
                            metadata: HashMap::new(),
                        };

                        let chunk = Chunk {
                            id: chunk_id.clone(),
                            file_path: rel_path.to_string(),
                            start_line,
                            end_line,
                            tokens_approx: section_content.split_whitespace().count(),
                            content: section_content,
                            summary: None,
                            node_id: Some(node_id.clone()),
                            vector_id: 0,
                        };

                        edges.push(Edge {
                            source: file_node_id.to_string(),
                            target: node_id.clone(),
                            kind: EdgeKind::Contains,
                            weight: 1.0,
                        });

                        if let Some(prev) = last_chunk_id {
                            edges.push(Edge {
                                source: prev,
                                target: chunk_id.clone(),
                                kind: EdgeKind::NextChunk,
                                weight: 0.5,
                            });
                        }
                        last_chunk_id = Some(chunk_id.clone());

                        nodes.push(node);
                        chunks.push(chunk);
                    }
                }
                current_section_start = i;
                current_title = caps[2].trim().to_string();
            }
        }

        // Final section
        if current_section_start < lines.len() {
            let section_lines = &lines[current_section_start..lines.len()];
            let section_content = section_lines.join("\n").trim().to_string();
            if !section_content.is_empty() {
                let start_line = current_section_start + 1;
                let end_line = lines.len();
                let chunk_id = format!("chunk:{}:{}:{}", rel_path, start_line, end_line);
                let node_id = format!("doc:{}:{}", rel_path, current_title);

                let node = Node {
                    id: node_id.clone(),
                    label: current_title.clone(),
                    kind: NodeKind::DocSection,
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    signature: None,
                    docstring: None,
                    chunk_id: Some(chunk_id.clone()),
                    vector_id: None,
                    metadata: HashMap::new(),
                };

                let chunk = Chunk {
                    id: chunk_id.clone(),
                    file_path: rel_path.to_string(),
                    start_line,
                    end_line,
                    tokens_approx: section_content.split_whitespace().count(),
                    content: section_content,
                    summary: None,
                    node_id: Some(node_id.clone()),
                    vector_id: 0,
                };

                edges.push(Edge {
                    source: file_node_id.to_string(),
                    target: node_id.clone(),
                    kind: EdgeKind::Contains,
                    weight: 1.0,
                });

                nodes.push(node);
                chunks.push(chunk);
            }
        }
    }

    // --------------------------------------------------------------------------------
    // Generic / Fallback Chunker (Lines window with overlap)
    // --------------------------------------------------------------------------------
    fn parse_generic(
        &self,
        rel_path: &str,
        content: &str,
        file_node_id: &str,
        chunks: &mut Vec<Chunk>,
        _nodes: &mut Vec<Node>,
        edges: &mut Vec<Edge>,
    ) {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return;
        }

        let chunk_size = 50;
        let overlap = 10;
        let mut i = 0;
        let mut last_chunk_id: Option<String> = None;

        while i < lines.len() {
            let end = (i + chunk_size).min(lines.len());
            let chunk_lines = &lines[i..end];
            let chunk_content = chunk_lines.join("\n");
            let start_line = i + 1;
            let end_line = end;
            let chunk_id = format!("chunk:{}:{}:{}", rel_path, start_line, end_line);

            let chunk = Chunk {
                id: chunk_id.clone(),
                file_path: rel_path.to_string(),
                start_line,
                end_line,
                tokens_approx: chunk_content.split_whitespace().count(),
                content: chunk_content,
                summary: None,
                node_id: None,
                vector_id: 0,
            };

            edges.push(Edge {
                source: file_node_id.to_string(),
                target: chunk_id.clone(),
                kind: EdgeKind::Contains,
                weight: 1.0,
            });

            if let Some(prev) = last_chunk_id {
                edges.push(Edge {
                    source: prev,
                    target: chunk_id.clone(),
                    kind: EdgeKind::NextChunk,
                    weight: 0.5,
                });
            }
            last_chunk_id = Some(chunk_id.clone());

            chunks.push(chunk);

            if end == lines.len() {
                break;
            }
            i += chunk_size - overlap;
        }
    }

    /// Recursively scan a folder and parse all supported files.
    pub fn scan_directory(&self, base_dir: &Path) -> anyhow::Result<Vec<ParsedFile>> {
        let mut results = Vec::new();

        for entry in WalkDir::new(base_dir)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                // Ignore build and hidden dirs (except current dir '.')
                let is_hidden = name.starts_with('.') && name != "." && name != "..";
                !is_hidden
                    && name != "target"
                    && name != "node_modules"
                    && name != "dist"
                    && name != "build"
            })
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() && self.is_supported(path) {
                if let Ok(rel) = path.strip_prefix(base_dir) {
                    let rel_str = rel.to_string_lossy().to_string();
                    if let Ok(parsed) = self.parse_file(path, &rel_str) {
                        results.push(parsed);
                    }
                }
            }
        }

        // Post-processing: infer call & reference edges across files
        self.infer_cross_symbol_edges(&mut results);

        Ok(results)
    }

    /// Infer call-shaped candidate edges; this pass does not resolve scopes or imports.
    fn infer_cross_symbol_edges(&self, files: &mut [ParsedFile]) {
        // Duplicate names remain ambiguous; do not silently pick one target.
        let mut symbol_map: HashMap<String, Vec<String>> = HashMap::new();
        for file in files.iter() {
            for node in &file.nodes {
                if node.kind == NodeKind::Function && node.label.len() >= 4 {
                    symbol_map
                        .entry(node.label.clone())
                        .or_default()
                        .push(node.id.clone());
                }
            }
        }

        let patterns: Vec<_> = symbol_map
            .into_iter()
            .filter_map(|(name, ids)| {
                Regex::new(&format!(r"\b{}\s*\(", regex::escape(&name)))
                    .ok()
                    .map(|pattern| (pattern, ids))
            })
            .collect();

        for file in files.iter_mut() {
            for chunk in &file.chunks {
                if let Some(ref source_node_id) = chunk.node_id {
                    if !file.nodes.iter().any(|n| {
                        n.id == *source_node_id
                            && matches!(n.kind, NodeKind::Function | NodeKind::Method)
                    }) {
                        continue;
                    }
                    for (pattern, target_ids) in &patterns {
                        if pattern.find_iter(&chunk.content).any(|hit| {
                            let prefix = chunk.content[..hit.start()]
                                .rsplit('\n')
                                .next()
                                .unwrap_or("")
                                .trim_end();
                            !prefix.ends_with("fn") && !prefix.ends_with("def")
                        }) {
                            for target_node_id in target_ids {
                                if source_node_id == target_node_id {
                                    continue;
                                }
                                file.edges.push(Edge {
                                    source: source_node_id.clone(),
                                    target: target_node_id.clone(),
                                    kind: EdgeKind::CallsCandidate,
                                    weight: 0.5,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn indexes_only_readable_text() {
        let dir =
            std::env::temp_dir().join(format!("vectordb-text-only-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.md"), "# Searchable notes\nActual text").unwrap();
        std::fs::write(dir.join("photo.png"), b"\x89PNG\r\n\x1a\nimage bytes").unwrap();
        std::fs::write(dir.join("hidden.bin.txt"), b"abc\0def").unwrap();
        std::fs::write(dir.join("office.docx"), b"PK\x03\x04data").unwrap();
        std::fs::write(dir.join("scan.pdf"), b"%PDF-1.4\n/image only").unwrap();
        std::fs::write(
            dir.join("paper.pdf"),
            b"%PDF-1.4\n(Extracted PDF paragraph)",
        )
        .unwrap();

        let parser = CodeParser::new();
        let files = parser.scan_directory(&dir).unwrap();
        let paths: HashSet<_> = files.iter().map(|file| file.file_path.as_str()).collect();
        assert_eq!(paths, HashSet::from(["notes.md", "paper.pdf"]));
        assert!(parser
            .parse_file(&dir.join("photo.png"), "photo.png")
            .is_err());
        assert!(parser
            .parse_file(&dir.join("scan.pdf"), "scan.pdf")
            .is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ambiguous_call_names_are_candidates_not_resolved_edges() {
        let dir = std::env::temp_dir().join(format!("vectordb-graph-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "pub fn resolve() {}\n").unwrap();
        std::fs::write(dir.join("b.rs"), "pub fn resolve() {}\n").unwrap();
        std::fs::write(dir.join("caller.rs"), "pub fn run() { resolve(); }\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "line\n".repeat(70)).unwrap();
        let files = CodeParser::new().scan_directory(&dir).unwrap();
        let caller = files.iter().find(|f| f.file_path == "caller.rs").unwrap();
        let candidates: Vec<_> = caller
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::CallsCandidate)
            .collect();
        assert_eq!(candidates.len(), 2);
        assert!(caller.edges.iter().all(|e| e.kind != EdgeKind::Calls));
        for definition in files
            .iter()
            .filter(|f| f.file_path == "a.rs" || f.file_path == "b.rs")
        {
            assert!(definition
                .edges
                .iter()
                .all(|e| e.kind != EdgeKind::CallsCandidate));
        }
        let engine = vectordb_core::VectorDBEngine::new(256);
        for file in &files {
            for node in &file.nodes {
                engine.add_graph_node(node.clone());
            }
        }
        for file in &files {
            for edge in &file.edges {
                engine.add_graph_edge(
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.kind.clone(),
                    edge.weight,
                );
            }
        }
        let node_ids: HashSet<_> = engine.get_all_nodes().into_iter().map(|n| n.id).collect();
        let edges = engine.get_all_edges();
        assert!(edges.iter().any(|e| e.kind == EdgeKind::NextChunk));
        assert!(edges
            .iter()
            .all(|e| node_ids.contains(&e.source) && node_ids.contains(&e.target)));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_parse_rust_code() {
        let code = r#"
/// Calculates the dot product of two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub struct HnswIndex {
    m: usize,
}
"#;
        let parser = CodeParser::new();
        let mut chunks = Vec::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        parser.parse_rust(
            "src/lib.rs",
            code,
            "file:src/lib.rs",
            &mut chunks,
            &mut nodes,
            &mut edges,
        );

        assert_eq!(chunks.len(), 2);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].label, "dot_product");
        assert_eq!(nodes[0].kind, NodeKind::Function);
        assert_eq!(nodes[1].label, "HnswIndex");
        assert_eq!(nodes[1].kind, NodeKind::Struct);
    }

    #[test]
    fn test_parse_rust_trait_impl_and_methods() {
        let code = r#"
pub trait Engine {
    fn search(&self, q: &str) -> Vec<String>;
}

pub struct MyEngine {
    dim: usize,
}

impl Engine for MyEngine {
    fn search(&self, q: &str) -> Vec<String> {
        vec![q.to_string()]
    }
}
"#;
        let parser = CodeParser::new();
        let mut chunks = Vec::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        parser.parse_rust(
            "src/engine.rs",
            code,
            "file:src/engine.rs",
            &mut chunks,
            &mut nodes,
            &mut edges,
        );

        // Should find Trait (Engine), Struct (MyEngine), Impl node (impl Engine for MyEngine), and Method (MyEngine::search)
        let labels: Vec<String> = nodes.iter().map(|n| n.label.clone()).collect();
        assert!(labels.contains(&"Engine".to_string()));
        assert!(labels.contains(&"MyEngine".to_string()));
        assert!(labels.contains(&"impl Engine for MyEngine".to_string()));
        assert!(labels.contains(&"MyEngine::search".to_string()));

        // Should have Implements edge
        let has_implements = edges.iter().any(|e| e.kind == EdgeKind::Implements);
        assert!(has_implements);
    }
}
