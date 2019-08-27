// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reading of PDF tokens and objects from a raw byte stream.

package main

import (
	"fmt"

	"github.com/rsc.io/pdf"
)

func main() {
	reader, _ := pdf.Open("./test.pdf")
	fmt.Printf("%v\n", reader.NumPage())
}
