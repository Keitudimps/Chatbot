use docx_rs::read_docx;
use std::fs::File;
use std::io::{BufReader, Read};

fn main() {
    println!("=== Extracting all document content from calendar_2024.docx ===\n");
    
    // Try multiple paths
    let paths = vec![
        "data/calendar_2024.docx",
        "./data/calendar_2024.docx",
        "calendar_2024.docx",
        "C:\\Users\\keitu\\Chatbot\\data\\calendar_2024.docx",
    ];
    
    let mut file_path = "";
    for path in &paths {
        if std::path::Path::new(path).exists() {
            file_path = path;
            println!("Found calendar at: {}\n", path);
            break;
        }
    }
    
    if file_path.is_empty() {
        eprintln!("Could not find calendar_2024.docx in any of: {:?}", paths);
        return;
    }
    
    let mut buffer = Vec::new();
    BufReader::new(File::open(file_path).expect("Failed to open"))
        .read_to_end(&mut buffer)
        .expect("Failed to read");

    let doc = read_docx(&buffer).expect("Failed to parse docx");
    
    // Extract all text from document
    let mut all_text = Vec::new();
    extract_text_from_document(&doc.document, &mut all_text);
    
    println!("=== All extracted text from document ===");
    for text in &all_text {
        if text.to_uppercase().contains("RECESS") || 
           text.to_uppercase().contains("GRADUATION") ||
           text.to_uppercase().contains("BREAK") ||
           text.to_uppercase().contains("VACATION") {
            println!(">>> {}", text);
        }
    }
}

fn extract_text_from_document(doc: &docx_rs::Document, texts: &mut Vec<String>) {
    use docx_rs::{DocumentChild, ParagraphChild, RunChild, TableChild, TableRowChild};
    
    for child in &doc.children {
        match child {
            DocumentChild::Paragraph(p) => {
                let mut text = String::new();
                for pc in &p.children {
                    if let ParagraphChild::Run(r) = pc {
                        for rc in &r.children {
                            if let RunChild::Text(t) = rc {
                                text.push_str(&t.text);
                            }
                        }
                    }
                }
                if !text.trim().is_empty() {
                    texts.push(text.trim().to_string());
                }
            }
            DocumentChild::Table(table) => {
                for row_child in &table.rows {
                    let TableChild::TableRow(row) = row_child;
                    for cell_child in &row.cells {
                        let TableRowChild::TableCell(cell) = cell_child;
                        for content in &cell.children {
                            if let docx_rs::TableCellContent::Paragraph(p) = content {
                                let mut text = String::new();
                                for pc in &p.children {
                                    if let ParagraphChild::Run(r) = pc {
                                        for rc in &r.children {
                                            if let RunChild::Text(t) = rc {
                                                text.push_str(&t.text);
                                            }
                                        }
                                    }
                                }
                                if !text.trim().is_empty() {
                                    texts.push(text.trim().to_string());
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
