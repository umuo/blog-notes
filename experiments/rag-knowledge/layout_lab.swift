import Foundation
import PDFKit
import Vision

struct Line: Codable {
    let text: String
    let x: Double
    let y: Double
}

struct PageEvidence: Codable {
    let page: Int
    let plainText: String
    let rows: [Line]
}

struct Cell: Codable {
    let field: String
    let meaning: String
    let page: Int
}

struct Result: Codable {
    let pages: [PageEvidence]
    let flatReadingOrder: [String]
    let columnReadingOrder: [String]
    let tableHeader: [String]
    let cells: [Cell]
    let missingValueQuestion: [String: Bool]
}

let location = URL(fileURLWithPath: CommandLine.arguments[1])
guard let document = PDFDocument(url: location) else {
    fatalError("cannot open PDF")
}
var pages: [PageEvidence] = []
for index in 0..<document.pageCount {
    guard let page = document.page(at: index),
          let data = page.thumbnail(of: CGSize(width: 1200, height: 900), for: .mediaBox).tiffRepresentation else {
        fatalError("cannot render page")
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = false
    try VNImageRequestHandler(data: data).perform([request])
    let lines = (request.results ?? []).compactMap { result -> Line? in
        guard let text = result.topCandidates(1).first?.string else {
            return nil
        }
        return Line(text: text, x: result.boundingBox.minX, y: result.boundingBox.midY)
    }.sorted { a, b in
        if abs(a.y - b.y) < 0.02 {
            return a.x < b.x
        }
        return a.y > b.y
    }
    pages.append(PageEvidence(page: index + 1, plainText: page.string ?? "", rows: lines))
}
guard pages.count == 2 else {
    fatalError("expected the two-page fixture")
}

let first = pages[0].rows
let paragraphs = first.filter { $0.y > 0.53 && $0.y < 0.89 }
let columnOrder = paragraphs.filter { $0.x < 0.5 } + paragraphs.filter { $0.x >= 0.5 }
let headers = first.filter { $0.y > 0.35 && $0.y < 0.47 }.map(\.text)
var cells: [Cell] = []
for page in pages {
    for left in page.rows where left.x < 0.35 && ["refund", "NA", "report_id"].contains(left.text) {
        guard let right = page.rows.first(where: { $0.x >= 0.35 && abs($0.y - left.y) < 0.025 }) else {
            continue
        }
        cells.append(Cell(field: left.text, meaning: right.text, page: page.page))
    }
}
let flatHasAnswer = pages.contains { $0.plainText.contains("missing value") }
let layoutHasAnswer = cells.contains { $0.field == "NA" && $0.meaning.lowercased() == "missing value" && $0.page == 2 }
precondition(!flatHasAnswer && layoutHasAnswer && headers == ["Field", "Meaning"])
precondition(columnOrder.map(\.text) != paragraphs.map(\.text))
let result = Result(pages: pages, flatReadingOrder: paragraphs.map(\.text), columnReadingOrder: columnOrder.map(\.text),
                    tableHeader: headers, cells: cells, missingValueQuestion: ["flat": flatHasAnswer, "layout": layoutHasAnswer])
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
print(String(decoding: try encoder.encode(result), as: UTF8.self))
