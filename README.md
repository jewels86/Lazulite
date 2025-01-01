# Lazulite
A modular, extensible framework that empowers developers to build advanced programming languages.

## Overview
Lazulite is a C#-powered framework for creating compilers for custom languages. 

## Development Progress
Lazulite is currently in development. Contributions are always welcome!

Currently, the following features are implemented:
- Tokenizer

### Test Language
The test langauge is a C-like langauge with the same syntax as C. The following code is an example of the test language:
```c
#include <sysio.h>

int main(int argc, char** argv) {
	printf("Hello, World!\n");

	int a = 5;
	int b = 10;
	int c = a + b;

	printf("The sum of %d and %d is %d\n", a, b, c);

	for (int i = 0; i < 10; i++) {
		printf("i = %d\n", i);
	}

	return 0;
}
```

## Todos
