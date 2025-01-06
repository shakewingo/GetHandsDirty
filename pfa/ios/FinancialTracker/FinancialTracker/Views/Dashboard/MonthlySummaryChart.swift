import SwiftUI
import Charts

struct MonthlySummaryChart: View {
    let data: [String: [String: Double]]
    
    private var chartData: [(month: String, category: String, amount: Double)] {
        var result: [(month: String, category: String, amount: Double)] = []
        
        for (month, categories) in data {
            for (category, amount) in categories {
                result.append((month: month, category: category, amount: amount))
            }
        }
        
        return result.sorted { $0.month < $1.month }
    }
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Monthly Summary")
                .font(.headline)
            
            Chart(chartData, id: \.month) { item in
                BarMark(
                    x: .value("Month", item.month),
                    y: .value("Amount", item.amount)
                )
                .foregroundStyle(by: .value("Category", item.category))
            }
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            .chartLegend(position: .bottom)
        }
    }
}