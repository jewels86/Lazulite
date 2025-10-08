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
    : LBRACE statement* RBRACE;
    
partialStatement 
    : RETURN expression
    | variableDeclaration;

statement 
    : partialStatement SEMICOLON;
    
operator
    : PLUS | MINUS | STAR | SLASH | PERCENT | CARET
    | PLUS PLUS | MINUS MINUS | SLASH SLASH;
    
unaryOperation 
    : IDENTIFIER operator;
    
binaryOperation 
    : IDENTIFIER operator IDENTIFIER;
    
parameter
    : (IDENTIFIER EQUAL)? expression;

parameterList
    : parameter (COMMA parameter)*;
    
functionCall
    : IDENTIFIER LPAREN parameterList? RPAREN;
    
expression
    : callExpression
    | LPAREN expression RPAREN;

callExpression
    : memberExpression
    | functionCall;

memberExpression
    : primaryExpression (DOT primaryExpression)*;

primaryExpression
    : literal
    | binaryOperation
    | unaryOperation
    | IDENTIFIER;

variableDeclaration
   :  LET IDENTIFIER (COLON modifier* type)? EQUAL expression;
   
declaredParameter
    : IDENTIFIER COLON modifier* type;
 
declaredParameterList
    : declaredParameter (COMMA declaredParameter);   

methodDeclaration 
    : IDENTIFIER LPAREN declaredParameterList? RPAREN INPLACE? ARROW modifier* type EQUAL block; 
    
literal
    : STRING | NUMBER;
    
operatorDeclaration 
    : OPERATOR methodDeclaration 
    | OPERATOR NEW LPAREN declaredParameterList? RPAREN INPLACE? ARROW modifier* type EQUAL block; 