# MCP ( Model Context Protocol )

- LSP ( Language Server Protocol ): 프로그래밍 언어와 IDE간의 통신 프로토콜 표준화
- MCP도 이와 같이 LLM과 도구 간의 통신 프로토콜을 표준화 한 것 

- MCP는 JSON-RPC 2.0을 기반으로 하며, stdio와 SSE, streamable-http 세가지 주요 통신 방식을 지원
- SSE는 긴 연결을 유지하기에 보안 우려사항이 있어 폐기 예정. 

```python
@mcp.tool()
async def sum_of_nums(nums:list[int]) -> int:
    """숫자 리스트의 총합을 반환한다."""
    return sum(nums)
```

## resource 

- LLM이 접근 할 수 있는 데이터를 의미

```python
@mcp.resource("dir://test")
def my_resource() -> List[str]:
    """test 폴더에 있는 파일 리스트"""
    test = Path.home() / "test"
    return [str(f) for f in test.iterdir()]
```

## prompt는 재사용 가능한 템플릿

```python
@mcp.prompt("코드 리뷰")
def sample_prompt(language: str):
    """특정 프로그램 언어에 대한 리뷰 프롬프트"""
    return f"""{language}에 대한 코드를 리뷰해주세요"""
```

## install MCP 

```sh
uv pip install "mcp[cli]"==1.16
uv pip show mcp
```

## Simple MCP Server 

- [code](simple_mcp_server.py). 

## MCP Test Tools 

- [inspector](https://github.com/modelcontextprotocol/inspector)
- [MCP Server](https://github.com/modelcontextprotocol/servers)
- [MCP remote](https://github.com/geelen/mcp-remote). 
- [MCP RoadMAP](https://modelcontextprotocol.io/development/roadmap) 
- [github discussion](https://github.com/orgs/modelcontextprotocol/discussions). 
- [Google A2A](https://github.com/a2aproject/A2A). 