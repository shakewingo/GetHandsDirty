import SwiftUI

struct BillsView: View {
    @StateObject private var viewModel = FinancialViewModel()
    @State private var isShowingUploadSheet = false
    
    var body: some View {
        NavigationView {
            VStack {
                Text("To Be Continued")
                    .font(.title)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationTitle("Bills")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        isShowingUploadSheet = true
                    }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $isShowingUploadSheet) {
                StatementUploadView(viewModel: viewModel)
            }
        }
        .alert("Error", isPresented: .constant(viewModel.errorMessage != nil)) {
            Button("OK") {
                viewModel.clearError()
            }
        } message: {
            if let errorMessage = viewModel.errorMessage {
                Text(errorMessage)
            }
        }
    }
}

struct StatementUploadView: View {
    @ObservedObject var viewModel: FinancialViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var selectedSourceType: String?
    
    let sourceTypes = [
        "td_chequing": "TD Chequing",
        "td_credit": "TD Credit",
        "cmb_chequing": "CMB Chequing",
        "cmb_credit": "CMB Credit"
    ]
    
    var body: some View {
        NavigationView {
            List {
                ForEach(Array(sourceTypes.keys.sorted()), id: \.self) { key in
                    Button(action: {
                        selectedSourceType = key
                    }) {
                        HStack {
                            Text(sourceTypes[key] ?? key)
                            Spacer()
                            if selectedSourceType == key {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select Statement Type")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    if let sourceType = selectedSourceType {
                        DocumentPicker(callback: { url in
                            viewModel.selectedFiles = [(url, sourceType)]
                            Task {
                                await viewModel.uploadSelectedFiles()
                                dismiss()
                            }
                        })
                    }
                }
            }
        }
    }
} 