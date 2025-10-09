parser grammar MRLParser;

options { tokenVocab = MRLLexer; }

program
    : declaration* EOF;
    
declaration
    : typeDeclaration
    | methodDeclaration
    | variableDeclaration SEMICOLON;
    
typeDeclaration
    : IDENTIFIER COLON (COMPLETE)? ITYPE interfaceList? (ALIKE identifierList)? EQUAL LBRACE memberDeclaration* RBRACE;

interfaceList
    : IDENTIFIER (COMMA IDENTIFIER)*;
identifierList
    : IDENTIFIER (COMMA IDENTIFIER)*;

memberDeclaration
    : fieldDeclaration
    | methodDeclaration
    | operatorDeclaration;
    
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
    : expression
    | LBRACE statement* RBRACE;
    
partialStatement 
    : RETURN expression
    | variableDeclaration
    | memberExpression EQUAL expression;
    
statement 
    : partialStatement SEMICOLON;
    
operator
    : PLUS | MINUS | STAR | SLASH | PERCENT | CARET
    | PLUS PLUS | MINUS MINUS | SLASH SLASH;
    
parameter
    : (IDENTIFIER EQUAL)? expression;

parameterList
    : parameter (COMMA parameter)*;
    
functionCall
    : IDENTIFIER LPAREN parameterList? RPAREN;
    
memberExpression 
    : (primaryExpression|functionCall) (DOT (primaryExpression|functionCall))*;
    
expression
    : functionCall
    | memberExpression
    | expression operator expression
    | operator expression
    | LPAREN expression RPAREN
    | withExpression;

primaryExpression
    : IDENTIFIER
    | literal;

variableDeclaration
   :  LET IDENTIFIER (COLON modifier* type)? EQUAL expression;
   
declaredParameter
    : IDENTIFIER COLON modifier* type;
    
nextDeclaredParameter
    : COMMA declaredParameter;
 
declaredParameterList
    : declaredParameter nextDeclaredParameter*;

methodDeclaration 
    : IDENTIFIER LPAREN declaredParameterList? RPAREN INPLACE? ARROW (modifier* type | PRESERVES IDENTIFIER) EQUAL block; 
    
literal
    : STRING | NUMBER;
    
operatorDeclaration 
    : OPERATOR IDENTIFIER operator LPAREN declaredParameterList? RPAREN INPLACE? ARROW (modifier* type | PRESERVES IDENTIFIER) EQUAL block
    | OPERATOR NEW LPAREN declaredParameterList? RPAREN INPLACE? EQUAL block; 
    
withExpression
    : primaryExpression WITH LBRACE (IDENTIFIER EQUAL expression) (COMMA IDENTIFIER EQUAL expression)* RBRACE;