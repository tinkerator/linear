package linear

import (
	"math"
	"testing"
)

func TestMXM(t *testing.T) {
	vs := []struct {
		a1, a2, ans, sna Matrix
	}{
		{
			a1: Matrix{
				{1, 2},
			},
			a2: Matrix{
				{3},
				{4},
			},
			ans: Matrix{
				{11},
			},
			sna: Matrix{
				{3, 6},
				{4, 8},
			},
		},
	}
	for i, v := range vs {
		b := v.a1.XM(v.a2)
		if !b.Same(v.ans) {
			t.Errorf("%d: got=%v, want=%v", i, b, v.ans)
		}
		t.Logf("%d: b =\n%v", i, b)
		if v.sna == nil {
			continue
		}
		c := v.a2.XM(v.a1)
		if !c.Same(v.sna) {
			t.Errorf("%d: got=%v, want=%v", i, c, v.sna)
		}
		t.Logf("%d: c =\n%v", i, c)
	}
}

func TestInv(t *testing.T) {
	a := Matrix{
		{1, 3},
		{2, 4},
	}
	recip := Matrix{
		{-2, 1.5},
		{1, -.5},
	}
	inv, err := a.Inv()
	if err != nil {
		t.Errorf("no inverse (%v) for\n%v", err, a)
	}
	if !inv.Same(recip) {
		t.Errorf("bad inverse =\n%v", inv)
	}
}

func TestFitPoly(t *testing.T) {
	xy := []Point{
		{-2.5, 18.25},
		{-1.5, 7.25},
		{-0.5, 2.25},
		{.5, 3.25},
		{1.5, 10.25},
		{2.5, 23.25},
		{3.5, 42.25},
		{4.5, 67.25},
		{5.5, 98.25},
		{6.5, 135.25},
		{7.5, 178.25},
		{8.5, 227.25},
		{9.5, 282.25},
	}
	fit, err := FitPoly(2, xy)
	if err != nil {
		t.Fatalf("unable to fit xy=%v: %v", xy, err)
	}
	expect := []float64{2, 1, 3}
	if len(fit) != len(expect) {
		t.Errorf("mismatch fit length: got=%v, want=%v", fit, expect)
	} else {
		for i, v := range fit {
			if math.Abs(v-expect[i]) > 0.001 {
				t.Errorf("mismatch fit: got=%v, want=%v", fit, expect)
				break
			}
		}
	}
	t.Logf("fit = %v", fit)
	for i, coord := range xy {
		y := fit.Expand(coord.X)
		if math.Abs(y-coord.Y) > 0.01 {
			t.Errorf("%d: got=%g want=%g", i, y, coord.Y)
		}
	}
}
