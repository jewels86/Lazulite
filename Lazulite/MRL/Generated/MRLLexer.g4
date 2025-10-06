lexer grammar MRLLexer;

ITYPE: 'itype';
COMPLETE: 'complete';
STATIC: 'static';
ISTATIC: 'istatic';
READONLY: 'readonly';
CONSTANT: 'constant';
ICONSTANT: 'iconstant';
DYNAMIC: 'dynamic';
REQUIRED: 'required';
NULLABLE: 'nullable';
SPECIFIC: 'specific';
INPLACE: 'inplace';
ALIKE: 'alike';
SAME: 'same';
OPERATOR: 'operator';
NEW: 'new';
WITH: 'with';
OUT: 'out';
PRESERVES: 'preserves';
FROM: 'from';
TO: 'to';
RETURN: 'return';
IMPORT: 'import';
ASSERT: 'assert';
IF: 'if';
ELSE: 'else';
NULL: 'null';
LET: 'let';
GET: 'get';
SET: 'set';
AS: 'as';
FOR: 'for';
IN: 'in';
WHILE: 'while';
BREAK: 'break';
CONTINUE: 'continue';

COLON: ':';
COMMA: ',';
DOT: '.';
SEMICOLON: ';';

ARROW: '->';
FULLARROW: '=>';

EQUAL: '=';
MODIFY: ':=';

LBRACE: '{';
RBRACE: '}';
LPAREN: '(';
RPAREN: ')';
LBRACK: '[';
RBRACK: ']';

PLUS: '+';
MINUS: '-';
STAR: '*';
SLASH: '/';
PERCENT: '%';
CARET: '^';
BANG: '!';
TILDE: '~';
QUESTION: '?';

LESSTHAN: '<';
LESSEQUAL: '<=';
GREATERTHAN: '>';
GREATEREQUAL: '>=';
EQUALEQUAL: '==';
NOTEQUAL: '!=';
AND: '&';
OR: '|';
ANDAND: '&&';
OROR: '||';

PLUSPLUS: '++';
MINUSMINUS: '--';
PLUSEQUAL: '+=';
MINUSEQUAL: '-=';
STAREQUAL: '*=';
SLASHEQUAL: '/=';
PERCENTEQUAL: '%=';

IDENTIFIER: [a-zA-Z_][a-zA-Z0-9_]*;
NUMBER: [0-9]+ ('.' [0-9]+)?(([eE][+-]? [0-9]+)?)?;
STRING : '"' (~["\\\r\n] | '\\' .)* '"';
LINECOMMENT: '#' ~[\r\n]* -> skip;
BLOCKCOMMENT: '/*' .*? '*/' -> skip;
WS: [ \t\r\n]+ -> skip;