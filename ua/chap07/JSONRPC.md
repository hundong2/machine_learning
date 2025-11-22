# JSON-RPC 상세 가이드

## 1. JSON-RPC란?

**JSON-RPC (JavaScript Object Notation Remote Procedure Call)**는 JSON(JavaScript Object Notation)을 사용하여 원격 프로시저 호출(RPC)을 수행하는 경량화된 프로토콜입니다.

### 주요 특징
- **단순성**: 사양이 매우 간단하여 구현이 쉽습니다.
- **전송 중립성 (Transport Agnostic)**: HTTP, WebSocket, TCP, IPC 등 다양한 전송 계층 위에서 사용할 수 있습니다.
- **무상태 (Stateless)**: 각 요청은 독립적이며, 서버는 클라이언트의 상태를 유지할 필요가 없습니다.

### 프로토콜 구조 (v2.0 기준)

#### 요청 (Request) 객체
클라이언트가 서버에 보내는 메시지입니다.
```json
{
  "jsonrpc": "2.0",     // 프로토콜 버전 (필수)
  "method": "subtract", // 호출할 메서드 이름 (필수)
  "params": [42, 23],   // 메서드 파라미터 (선택, 배열 또는 객체)
  "id": 1               // 요청 식별자 (필수, 응답 매칭용)
}
```

#### 응답 (Response) 객체
서버가 클라이언트에게 보내는 메시지입니다.
**성공 시:**
```json
{
  "jsonrpc": "2.0",
  "result": 19,         // 메서드 실행 결과
  "id": 1               // 요청의 id와 동일
}
```

**실패 시:**
```json
{
  "jsonrpc": "2.0",
  "error": {            // 에러 객체
    "code": -32601,
    "message": "Method not found"
  },
  "id": 1
}
```

#### 알림 (Notification)
응답이 필요 없는 요청입니다. `id` 필드가 없습니다.
```json
{
  "jsonrpc": "2.0",
  "method": "update",
  "params": [1, 2, 3]
}
```

---

## 2. 실무 활용 사례

1.  **블록체인 (Blockchain)**
    *   **Ethereum, Bitcoin**: 노드와 통신하기 위한 표준 API로 JSON-RPC를 사용합니다. 지갑 애플리케이션(Metamask 등)이 블록체인 네트워크에 트랜잭션을 보내거나 잔액을 조회할 때 사용됩니다.
    *   예: `eth_getBalance`, `eth_sendTransaction`

2.  **LSP (Language Server Protocol)**
    *   **VS Code**: IDE와 언어 서버(Python, C++, Java 등 분석기) 간의 통신에 JSON-RPC를 기반으로 한 프로토콜을 사용합니다. 이를 통해 하나의 언어 서버로 여러 IDE를 지원할 수 있습니다.

3.  **마이크로서비스 간 통신**
    *   REST보다 더 명시적인 "행동(Action)" 중심의 API가 필요할 때, 또는 gRPC를 도입하기엔 너무 무거울 때 간단한 내부 통신용으로 사용됩니다.

---

## 3. 언어별 사용 예제

### 3.1 Python (서버 및 클라이언트)

Python은 딕셔너리와 리스트가 JSON 구조와 1:1로 매핑되어 구현이 매우 직관적입니다.

**Server (간단한 HTTP 기반 구현)**
```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class JSONRPCServer(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. 요청 읽기
        length = int(self.headers["Content-Length"])
        request_body = self.rfile.read(length)
        request = json.loads(request_body)
        
        print(f"Received: {request}")

        # 2. 메서드 처리 (라우팅)
        response = {
            "jsonrpc": "2.0",
            "id": request.get("id")
        }

        try:
            if request["method"] == "add":
                # params가 리스트([a, b])라고 가정
                result = sum(request["params"])
                response["result"] = result
            else:
                raise ValueError("Method not found")
        except Exception as e:
            response["error"] = {"code": -32601, "message": str(e)}

        # 3. 응답 전송
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    server = HTTPServer(("localhost", 8080), JSONRPCServer)
    print("JSON-RPC Server running on port 8080...")
    server.serve_forever()
```

**Client**
```python
import requests
import json

url = "http://localhost:8080"
payload = {
    "jsonrpc": "2.0",
    "method": "add",
    "params": [10, 20],
    "id": 1
}

response = requests.post(url, json=payload)
print(f"Response: {response.json()}")
# 출력: {"jsonrpc": "2.0", "result": 30, "id": 1}
```

### 3.2 C# (Client)

.NET 환경에서는 `System.Text.Json`을 사용하여 객체를 직렬화하여 전송합니다.

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class Program
{
    static async Task Main()
    {
        using var client = new HttpClient();
        
        var request = new
        {
            jsonrpc = "2.0",
            method = "add",
            params = new[] { 10, 20 },
            id = 1
        };

        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        Console.WriteLine($"Sending: {json}");
        
        var response = await client.PostAsync("http://localhost:8080", content);
        var responseString = await response.Content.ReadAsStringAsync();

        Console.WriteLine($"Received: {responseString}");
    }
}
```

### 3.3 C++ (구조 예제)

C++은 `nlohmann/json` 같은 라이브러리를 사용하여 JSON을 다룹니다. 네트워킹은 `cpr`이나 `boost::asio`를 사용하지만, 여기서는 JSON 구조 생성에 집중합니다.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp> // https://github.com/nlohmann/json

using json = nlohmann::json;

int main() {
    // 1. 요청 객체 생성
    json request;
    request["jsonrpc"] = "2.0";
    request["method"] = "add";
    request["params"] = {10, 20}; // 배열 파라미터
    request["id"] = 1;

    std::string serialized_req = request.dump();
    std::cout << "Serialized Request: " << serialized_req << std::endl;

    // (여기서 HTTP POST 전송 수행...)

    // 2. 응답 파싱 예제
    std::string server_response = R"({"jsonrpc": "2.0", "result": 30, "id": 1})";
    
    try {
        json response = json::parse(server_response);
        if (response.contains("result")) {
            std::cout << "Result: " << response["result"] << std::endl;
        } else if (response.contains("error")) {
            std::cout << "Error: " << response["error"]["message"] << std::endl;
        }
    } catch (json::parse_error& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
    }

    return 0;
}
```

---

## 4. 관련 기술 및 최신 트렌드

JSON-RPC와 유사하거나 대체제로 사용되는 최신 기술들입니다.

### 1. gRPC (Google Remote Procedure Call)
*   **특징**: Google에서 개발한 고성능 RPC 프레임워크. JSON 대신 **Protocol Buffers**를 사용하여 바이너리 통신을 하므로 데이터 크기가 작고 속도가 매우 빠릅니다.
*   **용도**: 마이크로서비스 간 통신, 성능이 중요한 백엔드 시스템.
*   **URL**: [https://grpc.io/](https://grpc.io/)

### 2. GraphQL
*   **특징**: Facebook에서 개발한 쿼리 언어. 클라이언트가 필요한 데이터 구조를 정의해서 요청하면 서버가 그에 맞춰 응답합니다. Over-fetching(불필요한 데이터 수신) 문제를 해결합니다.
*   **용도**: 복잡한 데이터 요구사항을 가진 프론트엔드-백엔드 통신.
*   **URL**: [https://graphql.org/](https://graphql.org/)

### 3. tRPC
*   **특징**: TypeScript 환경에서 **End-to-End 타입 안전성**을 보장하는 RPC 프레임워크입니다. 스키마 선언 없이도 서버의 타입이 클라이언트로 추론됩니다.
*   **용도**: Next.js 등 풀스택 TypeScript 웹 개발.
*   **URL**: [https://trpc.io/](https://trpc.io/)

### 4. MCP (Model Context Protocol) 🔥 (최신 트렌드)
*   **특징**: AI 모델(LLM)과 외부 시스템(데이터, 도구)을 연결하기 위한 표준 프로토콜입니다. Anthropic 등에서 주도하며, AI 에이전트가 로컬 파일이나 원격 리소스에 접근하는 표준 방식을 정의합니다. JSON-RPC와 유사한 구조를 가질 수 있습니다.
*   **용도**: AI 에이전트 개발, LLM의 도구 사용(Tool Use) 표준화.
*   **URL**: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
