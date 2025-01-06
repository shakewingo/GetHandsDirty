import SwiftUI

struct DashboardView: View {
    @StateObject private var viewModel = FinancialViewModel()
    @State private var isShowingDocumentPicker = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Summary Cards
                    if let summary = viewModel.summary {
                        HStack(spacing: 16) {
                            ValueItem(
                                title: "Total Assets",
                                value: viewModel.formatCurrency(summary.totalAssets),
                                color: .green
                            )
                            
                            ValueItem(
                                title: "Total Credit",
                                value: viewModel.formatCurrency(summary.totalCredit),
                                color: .red
                            )
                        }
                        
                        ValueItem(
                            title: "Net Worth",
                            value: viewModel.formatCurrency(summary.netWorth),
                            color: summary.netWorth >= 0 ? .green : .red
                        )
                        
                        // Monthly Summary Chart
                        if !summary.monthlySummary.isEmpty {
                            MonthlySummaryChart(data: summary.monthlySummary)
                                .frame(height: 300)
                                .padding()
                        }
                    }
                    
                    // File Upload Section
                    MultipleDocumentPicker(selectedFiles: $viewModel.selectedFiles)
                    
                    if !viewModel.selectedFiles.isEmpty {
                        Button(action: {
                            Task {
                                await viewModel.uploadSelectedFiles()
                            }
                        }) {
                            if viewModel.isLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle())
                            } else {
                                Text("Upload \(viewModel.selectedFiles.count) Files")
                                    .bold()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isLoading)
                    }
                    
                    // Recent Transactions
                    if !viewModel.transactions.isEmpty {
                        VStack(alignment: .leading) {
                            Text("Recent Transactions")
                                .font(.headline)
                                .padding(.horizontal)
                            
                            ForEach(viewModel.transactions) { transaction in
                                TransactionRow(transaction: transaction)
                                    .padding(.horizontal)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Financial Dashboard")
            .refreshable {
                await viewModel.fetchSummary()
            }
        }
        .task {
            await viewModel.fetchSummary()
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

struct TransactionRow: View {
    let transaction: Transaction
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(transaction.description)
                .font(.headline)
            HStack {
                Text(transaction.date)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Text(formatAmount(transaction.amount))
                    .font(.subheadline)
                    .foregroundColor(transaction.amount >= 0 ? .green : .red)
            }
            Text(transaction.category)
                .font(.caption)
                .padding(4)
                .background(Color.secondary.opacity(0.2))
                .cornerRadius(4)
        }
        .padding(.vertical, 4)
    }
    
    private func formatAmount(_ amount: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "CAD"
        return formatter.string(from: NSNumber(value: amount)) ?? "$0.00"
    }
}

#Preview {
    DashboardView()
} 
