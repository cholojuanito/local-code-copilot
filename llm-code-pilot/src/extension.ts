// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import { Range } from 'vscode';

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('llm-code-pilot started');

    // Activate inline code suggestions
    const inlineProvider: vscode.InlineCompletionItemProvider = {
        async provideInlineCompletionItems(document, position, completion_context, token) {
            console.log('provideInlineCompletionItems triggered');

            const editor = vscode.window.activeTextEditor;
            const selection = editor?.selection;
            const manuallyTriggered = completion_context.triggerKind == vscode.InlineCompletionTriggerKind.Invoke;

            if (!manuallyTriggered) return [];

            // If highlighted from right to left, put cursor at end and retrigger
            if (selection && position.isEqual(selection.start)) {
                console.log('Changing highlight direction')
                editor!.selection = new vscode.Selection(selection.start, selection.end);
                vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
                return [];
            }

            // Grab any highlighted text and send to LLM for suggestions
            if (selection && !selection.isEmpty) {
                const selectionRange = new Range(selection.start, selection.end);

                const highlighted = editor.document.getText(selectionRange);

                console.log(highlighted)

                var payload = {
                    prompt: highlighted
                };

                console.log('Sending request to server')
                const response = await fetch(
                    'http://localhost:8000/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(payload)
                    }
                );
                var responseText = await response.text();
                responseText = responseText.replace(/\\r\\n/g, '\n')

                var range = new Range(selection.end, selection.end);

                return new Promise<vscode.InlineCompletionItem[]>((resolve) => {
                    console.log(responseText)
                    resolve([{ insertText: responseText, range: range }])
                });
            }
        },
    };

    // Register for python files
    vscode.languages.registerInlineCompletionItemProvider({ scheme: 'file', language: 'python' }, inlineProvider);
}


// This method is called when your extension is deactivated
export function deactivate() {}
