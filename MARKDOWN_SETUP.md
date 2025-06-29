# 📝 Markdown Viewing Setup for Cursor

## Built-in Preview (Already Available!)

### Quick Commands
- **`Cmd+K V`** (Mac) / **`Ctrl+K V`** (Windows) - Side-by-side preview
- **`Cmd+Shift+V`** (Mac) / **`Ctrl+Shift+V`** (Windows) - New tab preview

### How to Use
1. Open any `.md` file (like `README.md`)
2. Press `Cmd+K V` to split the view
3. Edit on the left, see preview on the right
4. Preview updates automatically as you type!

## Recommended Extensions

### 1. Markdown All in One
- **Extension ID**: `yzhang.markdown-all-in-one`
- **Features**: 
  - Table of contents
  - Auto-numbered lists
  - Math equations
  - Shortcuts for formatting

### 2. Markdown Preview Enhanced
- **Extension ID**: `shd101wyy.markdown-preview-enhanced`
- **Features**:
  - Better styling
  - Mermaid diagrams
  - Chart support
  - Export to PDF/HTML

### 3. markdownlint
- **Extension ID**: `davidanson.vscode-markdownlint`
- **Features**:
  - Markdown linting
  - Style consistency
  - Error highlighting

## Installation Methods

### Method 1: Command Palette
1. Press `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows)
2. Type: `Extensions: Install Extensions`
3. Search for the extension name
4. Click "Install"

### Method 2: Extensions Panel
1. Click the Extensions icon in the sidebar (4 squares)
2. Search for the extension
3. Click "Install"

### Method 3: Quick Install Commands
Open terminal in Cursor and run:

```bash
# Install via command line (if available)
code --install-extension yzhang.markdown-all-in-one
code --install-extension shd101wyy.markdown-preview-enhanced
code --install-extension davidanson.vscode-markdownlint
```

## Custom Settings

Add to your `settings.json` for better markdown experience:

```json
{
    "markdown.preview.breaks": true,
    "markdown.preview.fontSize": 14,
    "markdown.preview.lineHeight": 1.6,
    "markdown.preview.markEditorSelection": true,
    "markdown.preview.scrollPreviewWithEditor": true,
    "markdown.preview.scrollEditorWithPreview": true,
    "markdownlint.config": {
        "MD033": false,
        "MD041": false
    }
}
```

## Keyboard Shortcuts

### Built-in
- `Cmd+K V` - Open preview to the side
- `Cmd+Shift+V` - Open preview in new tab
- `Cmd+K Z` - Zen mode (distraction-free writing)

### With Markdown All in One
- `Cmd+B` - Bold text
- `Cmd+I` - Italic text
- `Cmd+Shift+]` - Toggle heading (up)
- `Cmd+Shift+[` - Toggle heading (down)

## Viewing Your Project Files

Perfect for viewing:
- `README.md` - Project documentation
- `WEB_DEPLOYMENT_GUIDE.md` - Deployment instructions
- `debug_flowchart.md` - Debug information
- Any other `.md` files in your project

## Pro Tips

1. **Pin the preview tab** - Right-click → "Pin Tab" to keep it open
2. **Sync scrolling** - Preview follows your cursor position
3. **Live updates** - Changes appear instantly in preview
4. **Print/Export** - Right-click preview → Print or use enhanced extensions for PDF export
5. **Theme matching** - Preview uses your editor theme 