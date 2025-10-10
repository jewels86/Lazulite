parser grammar MRLParser;

options { tokenVocab = MRLLexer; }

program
    : declaration* EOF;
    
declaration
    : typeDeclaration
    | methodDeclaration
    | variableDeclaration SEMICOLON;
    
typeDeclaration
    : IDENTIFIER COLON (COMPLETE)? ITYPE identifierList? (ALIKE identifierList)? EQUAL LBRACE memberDeclaration* RBRACE;

identifierList
    : IDENTIFIER (COMMA IDENTIFIER)*;

memberDeclaration
    : fieldDeclaration
    | methodSignature COMMA?
    | methodDeclaration COMMA?
    | operatorDeclaration COMMA?;
    
fieldDeclaration
    : IDENTIFIER COLON modifier* type (EQUAL initializer)? (COMMA)?;
    
modifier 
    : STATIC | ISTATIC | READONLY | CONSTANT | ICONSTANT | DYNAMIC | REQUIRED | NULLABLE | SPECIFIC | SAME;
    
type
    : IDENTIFIER (LBRACK RBRACK)*;
    
initializer 
    : expression
    | block
    | GET block (SET block)?;
    
block 
    : expression SEMICOLON
    | LBRACE statement* RBRACE;
    
partialStatement 
    : RETURN expression
    | variableDeclaration
    | callExpression assignmentOperator expression;
    
statement 
    : partialStatement SEMICOLON
    | ifStatement
    | foreachStatement;
    
operator
    : PLUS | MINUS | STAR | SLASH | PERCENT | CARET
    | PLUS PLUS | MINUS MINUS | SLASH SLASH | NEW;
    
assignmentOperator 
    : EQUAL | PLUS EQUAL | MINUS EQUAL | STAR EQUAL | SLASH EQUAL | MODIFY;
    
parameter
    : (IDENTIFIER EQUAL)? expression;

parameterList
    : parameter (COMMA parameter)*;
    
functionCall
    : IDENTIFIER LPAREN parameterList? RPAREN;

expression
    : assignmentExpression
    | lambdaExpression;

assignmentExpression
    : binaryExpression;

binaryExpression
    : unaryExpression (operator unaryExpression)*
    | LPAREN unaryExpression RPAREN (LPAREN unaryExpression RPAREN)*;

unaryExpression
    : operator unaryExpression
    | callExpression;

callExpression
    : primaryExpression
    | withExpression
    | functionCall
    | memberExpression;

memberExpression 
    : (primaryExpression | functionCall) (DOT (IDENTIFIER | functionCall))+;

primaryExpression
    : IDENTIFIER
    | literal
    | LPAREN expression RPAREN;

variableDeclaration
   :  LET IDENTIFIER (COLON modifier* type)? (EQUAL expression)?;
   
declaredParameter
    : IDENTIFIER COLON modifier* type;
    
nextDeclaredParameter
    : COMMA declaredParameter;
 
declaredParameterList
    : declaredParameter nextDeclaredParameter*;

methodSignature 
    : IDENTIFIER LPAREN declaredParameterList? RPAREN INPLACE? ARROW (modifier* type | PRESERVES IDENTIFIER);

methodDeclaration 
    : methodSignature EQUAL block;
    
literal
    : STRING | NUMBER | NULL | LBRACK (expression (COMMA expression)*)? RBRACK;
    
operatorDeclaration 
    : OPERATOR IDENTIFIER operator LPAREN declaredParameterList? RPAREN INPLACE? ARROW (modifier* type | PRESERVES IDENTIFIER) EQUAL block
    | OPERATOR NEW LPAREN declaredParameterList? RPAREN INPLACE? EQUAL block; 
    
comparisonOperator
    : EQUAL EQUAL | LESSTHAN | GREATERTHAN | LESSTHAN EQUAL | GREATERTHAN EQUAL | NOTEQUAL;
    
comparison
    : expression comparisonOperator expression;
    
withExpression
    : primaryExpression WITH LBRACE (IDENTIFIER EQUAL expression) (COMMA (IDENTIFIER EQUAL expression)?)* RBRACE;
    
foreachStatement 
    : FOR EACH IDENTIFIER IN expression (WHERE lambdaExpression)? block;
    
ifStatement
    : IF comparison block (ELSE (IF comparison)? block)?;
    
lambdaExpression
    : LPAREN parameterList RPAREN FULLARROW block;