import SwiftUI
import UniformTypeIdentifiers

struct MultipleDocumentPicker: View {
    @Binding var selectedFiles: [(URL, String)]  // (fileURL, sourceType)
    @State private var isShowingPicker = false
    @State private var currentSourceType: String = ""
    
    var body: some View {
        VStack(spacing: 20) {
            // Selected files list
            if !selectedFiles.isEmpty {
                List {
                    ForEach(selectedFiles, id: \.0) { file, sourceType in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(file.lastPathComponent)
                                    .font(.headline)
                                Text(sourceType)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Button(action: {
                                selectedFiles.removeAll { $0.0 == file }
                            }) {
                                Image(systemName: "trash")
                                    .foregroundColor(.red)
                            }
                        }
                    }
                }
                .frame(maxHeight: 200)
            }
            
            // Add files buttons
            VStack(spacing: 12) {
                Button(action: {
                    currentSourceType = "td_chequing"
                    isShowingPicker = true
                }) {
                    Label("Add TD Chequing Statement (PDF)", systemImage: "doc.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                
                Button(action: {
                    currentSourceType = "td_credit"
                    isShowingPicker = true
                }) {
                    Label("Add TD Credit Statement (CSV)", systemImage: "doc.text.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .padding()
        }
        .fileImporter(
            isPresented: $isShowingPicker,
            allowedContentTypes: currentSourceType == "td_chequing" ? [.pdf] : [.commaSeparatedText],
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                for url in urls {
                    // Add file if it's not already selected
                    if !selectedFiles.contains(where: { $0.0 == url }) {
                        selectedFiles.append((url, currentSourceType))
                    }
                }
            case .failure(let error):
                print("Error selecting files: \(error.localizedDescription)")
            }
        }
    }
}

#Preview {
    MultipleDocumentPicker(selectedFiles: .constant([]))
} 

