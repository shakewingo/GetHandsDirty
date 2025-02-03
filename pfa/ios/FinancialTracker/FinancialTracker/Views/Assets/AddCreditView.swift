import SwiftUI

struct AddCreditView: View {
    @ObservedObject var viewModel: FinancialViewModel
    @Environment(\.dismiss) private var dismiss
    
    @State private var creditType: String = ""
    @State private var marketValue: String = ""
    @State private var currency: String = "RMB"
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    private let currencies = ["RMB", "USD", "EUR", "CAD"]
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Credit Details")) {
                    TextField("Credit Type", text: $creditType)
                    
                    TextField("Market Value", text: $marketValue)
                        .keyboardType(.decimalPad)
                    
                    Picker("Currency", selection: $currency) {
                        ForEach(currencies, id: \.self) { currency in
                            Text(currency).tag(currency)
                        }
                    }
                }
            }
            .navigationTitle("Add Credit")
            .navigationBarItems(
                leading: Button("Cancel") {
                    dismiss()
                },
                trailing: Button("Save") {
                    saveCredit()
                }
            )
            .alert("Error", isPresented: $showAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(alertMessage)
            }
        }
    }
    
    private func saveCredit() {
        guard !creditType.isEmpty else {
            alertMessage = "Please enter a credit type"
            showAlert = true
            return
        }
        
        guard let value = Double(marketValue) else {
            alertMessage = "Please enter a valid market value"
            showAlert = true
            return
        }
        
        // Check if the value is positive and show warning
        guard value < 0 else {
            alertMessage = "Credit values should be negative. Please enter a negative amount."
            showAlert = true
            return
        }
        
        Task {
            await viewModel.addCredit(creditType: creditType, marketValue: value, currency: currency)
            dismiss()
        }
    }
} 