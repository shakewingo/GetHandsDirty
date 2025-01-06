import Foundation

enum APIError: Error {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case serverError(Int)
    case noData
    
    var localizedDescription: String {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let code):
            return "Server error with code: \(code)"
        case .noData:
            return "No data received from server"
        }
    }
}

class APIService {
    static let shared = APIService()
    
    #if targetEnvironment(simulator)
    private let baseURL = "http://localhost:8000/api"  // Use this for simulator
    #else
    private let baseURL = "http://127.0.0.1:8000/api"   // Use this for physical device
    #endif
    
    private init() {
        print("APIService initialized with baseURL: \(baseURL)")
    }
    
    func fetchSummary() async throws -> AccountSummary {
        let urlString = "\(baseURL)/summary"
        print("Attempting to fetch summary from: \(urlString)")
        
        guard let url = URL(string: urlString) else {
            print("Invalid URL: \(urlString)")
            throw APIError.invalidURL
        }
        
        do {
            print("Starting network request...")
            let (data, response) = try await URLSession.shared.data(from: url)
            print("Received response: \(response)")
            
            guard let httpResponse = response as? HTTPURLResponse else {
                print("Invalid response type")
                throw APIError.networkError(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
            }
            
            print("HTTP Status Code: \(httpResponse.statusCode)")
            guard (200...299).contains(httpResponse.statusCode) else {
                print("Error status code: \(httpResponse.statusCode)")
                throw APIError.serverError(httpResponse.statusCode)
            }
            
            // Debug: Print raw response
            if let jsonString = String(data: data, encoding: .utf8) {
                print("Raw API Response: \(jsonString)")
            }
            
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            
            let summary = try decoder.decode(AccountSummary.self, from: data)
            print("Successfully decoded summary: \(summary)")
            return summary
        } catch let error as DecodingError {
            print("Decoding Error: \(error)")
            throw APIError.decodingError(error)
        } catch {
            print("Network Error: \(error)")
            throw APIError.networkError(error)
        }
    }
    
    func uploadStatements(files: [(URL, String)]) async throws -> [Transaction] {
        guard let url = URL(string: "\(baseURL)/upload_statements") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var data = Data()
        
        // Add each file to the request
        for (fileURL, sourceType) in files {
            // Add file data
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"files\"; filename=\"\(fileURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
            data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
            data.append(try Data(contentsOf: fileURL))
            data.append("\r\n".data(using: .utf8)!)
            
            // Add source type
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"source_types\"\r\n\r\n".data(using: .utf8)!)
            data.append("\(sourceType)\r\n".data(using: .utf8)!)
        }
        
        data.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = data
        
        let (responseData, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
        }
        
        guard (200...299).contains(httpResponse.statusCode) else {
            throw APIError.serverError(httpResponse.statusCode)
        }
        
        let decoder = JSONDecoder()
        let result = try decoder.decode(UploadResponse.self, from: responseData)
        return result.transactions
    }
    
    struct UploadResponse: Codable {
        let transactions: [Transaction]
    }
}
