import SwiftUI

struct EditCreditView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel: FinancialViewModel
    let credit: Credit
    
    @State private var marketValue: String
    @State private var selectedCurrency: String
    @State private var createdAt: String
    @State private var showingAlert = false
    @State private var alertMessage = ""
    @State private var isLoading = false
    
    init(viewModel: FinancialViewModel, credit: Credit) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.credit = credit
        _marketValue = State(initialValue: String(credit.marketValue))
        _selectedCurrency = State(initialValue: credit.currency)
        _createdAt = State(initialValue: credit.createdAt)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Credit Details")) {
                    HStack {
                        Text("Type")
                        Spacer()
                        Text(credit.creditType)
                            .foregroundColor(.gray)
                    }
                    
                    TextField("Market Value", text: $marketValue)
                        .keyboardType(.decimalPad)
                    
                    Picker("Currency", selection: $selectedCurrency) {
                        Text("USD").tag("USD")
                        Text("EUR").tag("EUR")
                        Text("RMB").tag("RMB")
                        Text("CAD").tag("CAD")
                    }
                    
                    TextField("Created Date (YYYY-MM-DD)", text: $createdAt)
                        .keyboardType(.default)
                }
            }
            .navigationTitle("Edit Credit")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveCredit()
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
    
    private func saveCredit() {
        guard let value = Double(marketValue) else {
            alertMessage = "Please enter a valid market value"
            showingAlert = true
            return
        }
        
        isLoading = true
        
        Task {
            let updatedCredit = Credit(
                id: credit.id,
                creditType: credit.creditType,
                marketValue: value,
                currency: selectedCurrency,
                createdAt: createdAt
            )
            
            await viewModel.updateCredit(updatedCredit)
            
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