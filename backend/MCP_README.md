# Z-Image-Turbo MCP Server

This directory contains an MCP (Model Context Protocol) server implementation for the Z-Image-Turbo image generation model. The MCP server allows AI assistants and other clients to generate images through a standardized protocol.

## What is MCP?

MCP (Model Context Protocol) is an open protocol developed by Anthropic that standardizes how AI applications connect to external data sources and tools. It provides a secure, standardized way for AI assistants to access external capabilities.

## Features

The Z-Image-Turbo MCP server provides:

- **Image Generation Tool**: Generate images from text prompts with customizable parameters
- **Model Information Tool**: Get details about the loaded model and configuration
- **Configuration Management**: Update model settings dynamically
- **Example Prompts Resource**: Access curated example prompts and tips
- **Dual Transport Modes**: Support for both stdio and HTTP/SSE transports

## Installation

1. Install the required dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Ensure your `config.json` (in the project root) is properly configured:

```json
{
  "cache_dir": "./models",
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "cpu_offload": false
}
```

## Configuration

The MCP server can be configured via `mcp_config.json`:

```json
{
  "transport": "stdio",
  "host": "0.0.0.0",
  "port": 8001
}
```

### Configuration Options

- **transport**: Communication mode
  - `"stdio"`: For local clients (Claude Desktop, MCP Inspector)
  - `"streamable-http"`: For web clients and remote access
- **host**: Host address for HTTP mode (default: `"0.0.0.0"`)
- **port**: Port number for HTTP mode (default: `8001`)

## Running the MCP Server

### Method 1: Using the Shell Script (Recommended)

```bash
# Run with stdio (for Claude Desktop integration)
./run_mcp.sh --stdio

# Run with HTTP on default port 8001
./run_mcp.sh --http

# Run with HTTP on custom port
./run_mcp.sh --http --port 8080

# Run with HTTP on specific host and port
./run_mcp.sh --http --host 127.0.0.1 --port 9000
```

### Method 2: Direct Python Execution

```bash
# Using configuration from mcp_config.json
python mcp_server.py

# Override with command-line arguments
python mcp_server.py --transport stdio
python mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8001
```

## Available Tools

### 1. `generate_image`

Generate an image from a text prompt.

**Parameters:**
- `prompt` (string, required): Text description of the image to generate
- `width` (integer, optional): Image width in pixels (default: 1024)
- `height` (integer, optional): Image height in pixels (default: 1024)
- `num_inference_steps` (integer, optional): Number of denoising steps (default: 8, range: 1-50)
- `guidance_scale` (float, optional): Classifier-free guidance scale (default: 0.0, range: 0.0-20.0)
- `seed` (integer, optional): Random seed for reproducibility

**Returns:** Base64-encoded PNG image data

**Example:**
```json
{
  "prompt": "A serene mountain landscape at sunset",
  "width": 1024,
  "height": 768,
  "num_inference_steps": 8,
  "guidance_scale": 0.0,
  "seed": 42
}
```

### 2. `get_model_info`

Get information about the currently loaded model.

**Parameters:** None

**Returns:** Dictionary with model configuration and status

**Example Response:**
```json
{
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "cache_dir": "./models",
  "cpu_offload": false,
  "device": "cuda",
  "is_loaded": true,
  "cuda_available": true,
  "default_width": 1024,
  "default_height": 1024,
  "default_steps": 8,
  "default_guidance_scale": 0.0
}
```

### 3. `update_model_config`

Update model configuration settings (requires model reload).

**Parameters:**
- `cache_dir` (string, optional): Path to model cache directory
- `cpu_offload` (boolean, optional): Enable CPU offload for memory optimization

**Returns:** Status and updated configuration

### 4. Resource: `image://examples`

Access example prompts and tips for image generation.

**Returns:** JSON with example prompts and generation tips

## Integration Examples

### Claude Desktop Integration

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "python",
      "args": ["/path/to/z-image-turbo/backend/mcp_server.py", "--transport", "stdio"],
      "env": {}
    }
  }
}
```

Or using the shell script:

```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "/path/to/z-image-turbo/backend/run_mcp.sh",
      "args": ["--stdio"],
      "env": {}
    }
  }
}
```

### HTTP/SSE Integration

For web clients, start the server in HTTP mode:

```bash
./run_mcp.sh --http --port 8001
```

Then connect to `http://localhost:8001/mcp` using an MCP-compatible HTTP client.

### Python Client Example

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py", "--transport", "stdio"],
    env=None
)

# Connect and use
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # Generate an image
        result = await session.call_tool(
            "generate_image",
            arguments={
                "prompt": "A magical forest with glowing mushrooms",
                "width": 1024,
                "height": 1024
            }
        )

        # result contains base64-encoded image
        print(result)
```

## Troubleshooting

### Model Loading Issues

If the model fails to load:
1. Check that you have enough disk space in the cache directory
2. Verify internet connectivity for initial download
3. For GPU issues, ensure CUDA is properly installed
4. Enable `cpu_offload` in config.json if running out of GPU memory

### Port Already in Use

If port 8001 is already in use:
```bash
# Use a different port
./run_mcp.sh --http --port 8002
```

### Stdio Communication Issues

- Ensure no other output is written to stdout (only stderr logging is used)
- Check that the client is using the correct command and arguments
- Verify that dependencies are installed in the Python environment being used

## Performance Tips

1. **First Generation**: The first image generation will be slower as it loads the model
2. **Batch Processing**: Keep the server running to avoid model reload overhead
3. **GPU Usage**: Use CUDA if available for significantly faster generation
4. **CPU Offload**: Enable for large models on limited GPU memory
5. **Inference Steps**: Use 8 steps (default) for best speed/quality balance

## Security Considerations

- **HTTP Mode**: Only expose on trusted networks or use proper authentication
- **Resource Limits**: The server loads large AI models; ensure adequate system resources
- **Input Validation**: Prompts are processed as-is; implement content filtering if needed

## Development

### Project Structure

```
backend/
├── mcp_server.py        # Main MCP server implementation
├── mcp_config.json      # MCP server configuration
├── run_mcp.sh          # Convenience startup script
├── main.py             # Original FastAPI server
├── requirements.txt    # Python dependencies
└── MCP_README.md       # This file
```

### Adding New Tools

To add new tools to the MCP server, add functions decorated with `@mcp.tool()` in `mcp_server.py`:

```python
@mcp.tool()
async def my_new_tool(param: str) -> str:
    """
    Tool description (becomes the tool's description in MCP).

    Args:
        param: Parameter description

    Returns:
        Return value description
    """
    # Tool implementation
    return f"Result: {param}"
```

### Testing

Test the server using the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python mcp_server.py --transport stdio
```

## License

This MCP server implementation is part of the Z-Image-Turbo project.

## Support

For issues and questions:
- Check the main project README
- Review MCP documentation at https://modelcontextprotocol.io
- Open an issue on the project repository
