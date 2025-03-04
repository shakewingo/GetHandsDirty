import SwiftUI

struct AddAssetView: View {
    @ObservedObject var viewModel: FinancialViewModel
    @Environment(\.dismiss) private var dismiss
    
    @State private var assetType: String = ""
    @State private var marketValue: String = ""
    @State private var marketShare: String = ""
    @State private var currency: String = "RMB"
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    private let currencies = ["RMB", "USD", "EUR", "CAD"]
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Details")) {
                    TextField("Asset Type (e.g. AAPL)", text: $assetType)
                    
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
                }
                
                if !marketValue.isEmpty && !marketShare.isEmpty {
                    Text("Please fill only one: Market Value or Market Share")
                        .foregroundColor(.red)
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
            if !marketValue.isEmpty {
                if let value = Double(marketValue) {
                    await viewModel.addAsset(
                        assetType: assetType,
                        marketValue: value,
                        marketShare: nil,
                        currency: currency
                    )
                    dismiss()
                }
            } else if !marketShare.isEmpty {
                if let shares = Double(marketShare) {
                    await viewModel.addAsset(
                        assetType: assetType,
                        marketValue: nil,
                        marketShare: shares,
                        currency: currency
                    )
                    dismiss()
                }
            }
        }
    }
}

#Preview {
    AddAssetView(viewModel: FinancialViewModel())
} 
