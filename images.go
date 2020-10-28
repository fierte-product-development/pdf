// ただのメモ

package pdf

import (
	"fmt"

	"github.com/rsc.io/pdf/core"
)

func main() {
	r, _ := core.Open("./gazou.pdf")
	for i := 1; i <= r.NumPage(); i++ {
		p := r.Page(i)
		xobjs := p.Resources().Key("XObject")
		for _, k := range xobjs.Keys() {
			xobj := xobjs.Key(k)
			if xobj.Key("Subtype").Name() == "Image" {
				fmt.Printf("%v\n", xobj.Reader())
			}
		}
	}
}
