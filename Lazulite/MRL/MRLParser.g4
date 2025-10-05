parser grammar MRLParser;

options { tokenVocab = MRLLexer; }

program
    : declaration* EOF
    ;
    
declaration
    : typeDeclaration
    | functionDeclaration
    | variableDeclaration;
    
typeDeclaration
    : IDENTIFIER COLON (COMPLETE)? ITYPE interfaceList? (ALIKE identifierList)? EQUAL LBRACE memberDeclaration* RBRACE;

interfaceList
    : IDENTIFIER (COMMA IDENTIFIER)*;
identifierList
    : IDENTIFIER (COMMA IDENTIFIER)*;

memberDeclaration
    : fieldDeclaration
    | methodDeclaration;
    
fieldDeclaration
    : IDENTIFIER COLON modifier* type (EQUAL initializer)? (COMMA)?;
    
modifier 
    : STATIC | ISTATIC | READONLY | CONSTANT | ICONSTANT | DYNAMIC | REQUIRED | NULLABLE | SPECIFIC;
    
type
    : IDENTIFIER (LBRACK RBRACK)*;
    
initializer 
    : expression
    | block
    | GET block (SET block)?;
    
block 
    : LBRACE statement* RBRACE;
    
partialStatement 
    : RETURN expression;
    : 

statement 
    : partialStatement SEMICOLON;
    
operator
    : PLUS | MINUS | STAR | SLASH | PERCENT | CARET
    | PLUS PLUS | MINUS MINUS | SLASH SLASH;
    
unaryOperation 
    : IDENIFIER operator;
    
binaryOperation 
    : IDENTIFIER operator IDENTIFIER;
    
expression 
    : unaryOperation
    | binaryOperation
    | LPAREN expression RPAREN