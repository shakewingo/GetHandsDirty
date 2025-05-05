import SwiftUI

struct EditAssetView: View {
    @ObservedObject var viewModel: FinancialViewModel
    @Environment(\.dismiss) private var dismiss
    
    let asset: Asset
    
    @State private var assetType: String
    @State private var marketValue: String
    @State private var marketShare: String
    @State private var currency: String
    @State private var createdAt: String
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    private let currencies = ["RMB", "USD", "EUR", "CAD"]
    
    init(viewModel: FinancialViewModel, asset: Asset) {
        self.viewModel = viewModel
        self.asset = asset
        _assetType = State(initialValue: asset.assetType)
        _marketValue = State(initialValue: asset.marketValue.map { String($0) } ?? "")
        _marketShare = State(initialValue: asset.marketShare.map { String($0) } ?? "")
        _currency = State(initialValue: asset.currency)
        _createdAt = State(initialValue: asset.createdAt)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Details")) {
                    TextField("Asset Type (e.g. Gold, AAPL)", text: $assetType)
                    
                    Section(header: Text("Fill Either Value or Shares")) {
                        TextField("Market Value", text: $marketValue)
                            .keyboardType(.decimalPad)
                        
                        TextField("Market Share (# of shares)", text: $marketShare)
                            .keyboardType(.decimalPad)
                    }
                    
                    Picker("Currency", selection: $currency) {
                        ForEach(currencies, id: \.self) { currency in
                            Text(currency).tag(currency)
                        }
                    }
                    
                    TextField("Created Date (YYYY-MM-DD)", text: $createdAt)
                        .keyboardType(.default)
                }
                
                if !marketValue.isEmpty && !marketShare.isEmpty {
                    Text("Please fill only one: Market Value or Market Share")
                        .foregroundColor(.red)
                }
            }
            .navigationTitle("Edit Asset")
            .navigationBarItems(
                leading: Button("Cancel") {
                    dismiss()
                },
                trailing: Button("Save") {
                    saveAsset()
                }
                .disabled(assetType.isEmpty || 
                         (marketValue.isEmpty && marketShare.isEmpty) ||
                         (!marketValue.isEmpty && !marketShare.isEmpty))
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
        
        // Check that exactly one of marketValue or marketShare is filled
        if marketValue.isEmpty && marketShare.isEmpty {
            alertMessage = "Please enter either market value or market share"
            showAlert = true
            return
        }
        
        if !marketValue.isEmpty && !marketShare.isEmpty {
            alertMessage = "Please enter only one: market value or market share"
            showAlert = true
            return
        }
        
        Task {
            let updatedAsset = Asset(
                id: asset.id,
                assetType: assetType,
                marketValue: marketValue.isEmpty ? nil : Double(marketValue),
                marketShare: marketShare.isEmpty ? nil : Double(marketShare),
                currency: currency,
                createdAt: createdAt
            )
            await viewModel.updateAsset(updatedAsset)
            dismiss()
        }
    }
}

#Preview {
    EditAssetView(
        viewModel: FinancialViewModel(),
        asset: Asset(
            id: 1,
            assetType: "AAPL",
            marketValue: 150.0,
            marketShare: nil,
            currency: "USD",
            createdAt: "2023-01-01"
        )
    )
} 