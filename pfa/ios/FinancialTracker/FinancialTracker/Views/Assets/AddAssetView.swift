import SwiftUI

struct AddAssetView: View {
    @ObservedObject var viewModel: FinancialViewModel
    @Environment(\.dismiss) private var dismiss
    
    @State private var assetType: String = ""
    @State private var marketValue: String = ""
    @State private var currency: String = "RMB"
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    private let currencies = ["RMB", "USD", "EUR", "CAD"]
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Details")) {
                    TextField("Asset Type", text: $assetType)
                    
                    TextField("Market Value", text: $marketValue)
                        .keyboardType(.decimalPad)
                    
                    Picker("Currency", selection: $currency) {
                        ForEach(currencies, id: \.self) { currency in
                            Text(currency).tag(currency)
                        }
                    }
                }
            }
            .navigationTitle("Add Asset")
            .navigationBarItems(
                leading: Button("Cancel") {
                    dismiss()
                },
                trailing: Button("Save") {
                    saveAsset()
                }
            )
            .alert("Error", isPresented: $showAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(alertMessage)
            }
        }
    }
    
    private func saveAsset() {
        guard !assetType.isEmpty else {
            alertMessage = "Please enter an asset type"
            showAlert = true
            return
        }
        
        guard let value = Double(marketValue) else {
            alertMessage = "Please enter a valid market value"
            showAlert = true
            return
        }
        
        // Check if the value is negative and show warning
        guard value > 0 else {
            alertMessage = "Asset values should be positive. Please enter a positive amount."
            showAlert = true
            return
        }
        
        Task {
            await viewModel.addAsset(assetType: assetType, marketValue: value, currency: currency)
            dismiss()
        }
    }
}

#Preview {
    AddAssetView(viewModel: FinancialViewModel())
} 
