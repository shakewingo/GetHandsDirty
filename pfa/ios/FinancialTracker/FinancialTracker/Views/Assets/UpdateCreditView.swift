import SwiftUI

struct UpdateCreditView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var viewModel: FinancialViewModel
    let credit: Credit
    
    @State private var marketValue: String = ""
    @State private var currency: String = "EUR"
    @State private var showError = false
    
    private let currencies = ["EUR", "USD", "GBP"]
    
    init(viewModel: FinancialViewModel, credit: Credit) {
        self.viewModel = viewModel
        self.credit = credit
        _marketValue = State(initialValue: String(credit.marketValue))
        _currency = State(initialValue: credit.currency)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Credit Details")) {
                    Text("Type: \(credit.creditType)")
                        .foregroundColor(.gray)
                    
                    TextField("Market Value", text: $marketValue)
                        .keyboardType(.decimalPad)
                    
                    Picker("Currency", selection: $currency) {
                        ForEach(currencies, id: \.self) { currency in
                            Text(currency).tag(currency)
                        }
                    }
                }
            }
            .navigationTitle("Update Credit")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        save()
                    }
                }
            }
            .alert("Error", isPresented: $showError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage ?? "Unknown error occurred")
            }
        }
    }
    
    private func save() {
        guard let value = Double(marketValue) else {
            viewModel.errorMessage = "Invalid market value"
            showError = true
            return
        }
        
        let updatedCredit = Credit(
            id: credit.id,
            creditType: credit.creditType,
            marketValue: value,
            currency: currency,
            createdAt: credit.createdAt
        )
        
        Task {
            await viewModel.updateCredit(updatedCredit)
            if viewModel.errorMessage == nil {
                dismiss()
            } else {
                showError = true
            }
        }
    }
} 
