import SwiftUI

struct EditAssetView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel: FinancialViewModel
    let asset: Asset
    
    @State private var marketValue: String
    @State private var selectedCurrency: String
    @State private var showingAlert = false
    @State private var alertMessage = ""
    @State private var isLoading = false
    
    init(viewModel: FinancialViewModel, asset: Asset) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.asset = asset
        _marketValue = State(initialValue: String(asset.marketValue))
        _selectedCurrency = State(initialValue: asset.currency)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Asset Details")) {
                    HStack {
                        Text("Type")
                        Spacer()
                        Text(asset.assetType)
                            .foregroundColor(.gray)
                    }
                    
                    TextField("Market Value", text: $marketValue)
                        .keyboardType(.decimalPad)
                    
                    Picker("Currency", selection: $selectedCurrency) {
                        Text("USD").tag("USD")
                        Text("EUR").tag("EUR")
                        Text("RMB").tag("RMB")
                    }
                }
            }
            .navigationTitle("Edit Asset")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveAsset()
                    }
                    .disabled(isLoading)
                }
            }
            .alert("Error", isPresented: $showingAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(alertMessage)
            }
        }
        .interactiveDismissDisabled()
    }
    
    private func saveAsset() {
        guard let value = Double(marketValue) else {
            alertMessage = "Please enter a valid market value"
            showingAlert = true
            return
        }
        
        isLoading = true
        
        Task {
            let updatedAsset = Asset(
                id: asset.id,
                assetType: asset.assetType,
                marketValue: value,
                currency: selectedCurrency,
                createdAt: asset.createdAt
            )
            
            await viewModel.updateAsset(updatedAsset)
            
            // Only dismiss if there was no error
            if viewModel.errorMessage == nil {
                dismiss()
            } else {
                alertMessage = viewModel.errorMessage ?? "Unknown error occurred"
                showingAlert = true
                viewModel.clearError() // Clear the error so it doesn't trigger the parent view's alert
            }
            
            isLoading = false
        }
    }
} 