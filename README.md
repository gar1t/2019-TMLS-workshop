# Clean Coding for ML Workshop

## Topics to Discuss

- Naming conventions
- Notebooks
- Classes and objects
- Functions
- Command line interfaces
- Automation
- Linting
- Factoring working code
  - Step 1 - get it working
  - Step 2 - respell it based on understanding and intent
    - Look for functions
    - Avoid shared state
- Patterns
  - Apply
  - Try
  - Maybe
  - Dispatch
  - Find apply
  - Objects as struct / no methods

## Some Principles

- Be intentional
- Strive for algorithmic clarity
- Minimize context
- Minimize side-effects

## Notes - General

- Understand the nature of a language and don't work against
  it. E.g. Python is not a strong typed language. Even type hints
  don't make the language strongly typed. Python is not a pure
  language. Python is not a functional language, entirely. Object
  methods are common in the core API. Don't write code that looks
  different from other code in that language. Unless that language is
  Java.

- Variable and argument names are easier in smaller functions. The
  function provides context.

- A good metric for when you're "done" with a function is its
  length. Except in data science, where building plots can be a
  lengthy process. These cases make function length harder to use as a
  single metric. In general, one function per plot or transform.

- Functions name blocks of code and isolate their interface. You'd be
  surprised at a) how hard it is at times to name a function and b)
  how hard it is to find the right boundary of inputs and
  outputs. This is all excellent pain to experience because it's the
  process of understanding what you're doing and codifying that for
  yourself and others to see and use later.

- Tests are over rated. Don't feel bad for not having 100% test
  coverage. If your program needs 100% test coverage, don't use
  Python.

## Continue Discussion on Slack

To continue the workshop discussion, we started a Slack workspace.

Here's the sign up link - open to all!

[http://bit.ly/ml-engineering-slack](http://bit.ly/ml-engineering-slack)
