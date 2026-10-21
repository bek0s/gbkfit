// The colours of YAML in the editor, by the tokens of the page (see
// styles.css): keys, values, quoted strings, comments and punctuation.
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { tags } from '@lezer/highlight';

export const yamlStyle = syntaxHighlighting(
  HighlightStyle.define([
    { tag: tags.definition(tags.propertyName), color: 'var(--key)' },
    { tag: tags.content, color: 'var(--num)' },
    { tag: [tags.string, tags.special(tags.string)], color: 'var(--str)' },
    { tag: tags.lineComment, color: 'var(--com)', fontStyle: 'italic' },
    { tag: [tags.separator, tags.brace, tags.squareBracket, tags.punctuation], color: 'var(--faint)' },
    { tag: [tags.labelName, tags.typeName, tags.keyword, tags.meta], color: 'var(--accent)' },
  ]),
);
