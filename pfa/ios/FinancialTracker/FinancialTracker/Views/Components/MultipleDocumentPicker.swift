import SwiftUI
import UniformTypeIdentifiers

struct MultipleDocumentPicker: View {
    @Binding var selectedFiles: [(URL, String)]  // (fileURL, sourceType)
    @State private var isShowingPicker = false
    @State private var currentSourceType: String = ""
    @State private var isShowingSourceTypePicker = false
    
    let sourceTypes = [
        "td_chequing": "TD Chequing",
        "td_credit": "TD Credit",
        "cmb_chequing": "CMB Chequing",
        "cmb_credit": "CMB Credit"
    ]
    
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
                                Text(sourceTypes[sourceType] ?? sourceType)
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
            
            // Action buttons
            VStack(spacing: 12) {
                // Upload button
                Button(action: {
                    isShowingSourceTypePicker = false
                    isShowingPicker = true
                }) {
                    Label("Upload Statements", systemImage: "doc.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                
                // Type selection button
                Button(action: {
                    isShowingSourceTypePicker = true
                }) {
                    Label("Select Type", systemImage: "list.bullet")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                
                // Type selection menu
                if isShowingSourceTypePicker {
                    VStack(spacing: 8) {
                        ForEach(Array(sourceTypes.keys.sorted()), id: \.self) { key in
                            Button(action: {
                                currentSourceType = key
                                isShowingPicker = true
                                isShowingSourceTypePicker = false
                            }) {
                                Text(sourceTypes[key] ?? key)
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    .transition(.scale)
                }
            }
            .padding()
            .animation(.easeInOut, value: isShowingSourceTypePicker)
        }
        .fileImporter(
            isPresented: $isShowingPicker,
            allowedContentTypes: [.pdf, .commaSeparatedText],
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

