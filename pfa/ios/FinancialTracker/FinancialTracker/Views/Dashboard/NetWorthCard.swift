import SwiftUI

struct NetWorthCard: View {
    let summary: AccountSummary?
    @StateObject private var viewModel = FinancialViewModel()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Net Worth")
                .font(.headline)
            
            HStack(spacing: 20) {
                ValueItem(title: "Assets", value: viewModel.formatCurrency(summary?.totalAssets ?? 0), color: .green)
                ValueItem(title: "Credit", value: viewModel.formatCurrency(summary?.totalCredit ?? 0), color: .red)
                ValueItem(title: "Net", value: viewModel.formatCurrency(summary?.netWorth ?? 0), color: .blue)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
} 