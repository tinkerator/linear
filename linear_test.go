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
		if v.sna == nil {
			continue
		}
		c := v.a2.XM(v.a1)
		if !c.Same(v.sna) {
			t.Errorf("%d: got=%v, want=%v", i, c, v.sna)
		}
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
	ans := inv.XM(a)
	if math.Abs(ans[0][0]-1) > Zeroish || math.Abs(ans[1][1]-1) > Zeroish || math.Abs(ans[0][1]) > Zeroish || math.Abs(ans[1][0]) > Zeroish {
		t.Errorf("ans=%v is not identity", ans)
	}
}

func TestBinomial(t *testing.T) {
	vs := []struct {
		n, i int
		want float64
	}{
		{n: 1, i: 0, want: 1},
		{n: 2, i: 1, want: 2},
		{n: 4, i: 1, want: 4},
	}
	for i, v := range vs {
		got := Binomial(v.n, v.i)
		if math.Abs(got-v.want) > Zeroish {
			t.Errorf("%d: binomial(%d,%d) got=%f want=%f", i, v.n, v.i, v.want, got)
		}
	}
}

func TestFitSmall(t *testing.T) {
	xy := []Point{
		{-32, -5.5},
		{0, -3.2},
		{32, -5},
	}
	fit, err := FitPoly(2, xy)
	if err != nil {
		t.Fatalf("unable to fit xy=%v: %v", xy, err)
	}
	for i, coord := range xy {
		got := fit.Expand(coord.X)
		if math.Abs(got-coord.Y) > Zeroish {
			t.Errorf("%d: got=%f want %f", i, got, coord.Y)
		}
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
	for i, coord := range xy {
		y := fit.Expand(coord.X)
		if math.Abs(y-coord.Y) > 0.01 {
			t.Errorf("%d: got=%g want=%g", i, y, coord.Y)
		}
	}
}

func TestAffine(t *testing.T) {
	triangle := []Point{
		{0, 0},
		{3, 0},
		{5, 7},
	}
	for i := 0; i < 10; i++ {
		ang := float64(i*15) / 180 * math.Pi
		c, s := math.Cos(ang), math.Sin(ang)
		var v []Point
		sx, sy := 1+ang/math.Pi, 2-ang/math.Pi
		for _, pt := range triangle {
			v = append(v, Point{
				X: c*pt.X*sx - s*pt.Y*sy + ang,
				Y: s*pt.X*sx + c*pt.Y*sy - ang/2,
			})
		}
		aff, err := DeriveAffine(triangle, v)
		if err != nil {
			t.Errorf("[%d] failed to generate Affine for %v -> %v: %v", i, triangle, v, err)
			continue
		}
		inv, err := aff.Inv()
		if err != nil {
			t.Fatalf("inverse of affine not defined: aff=%#f: %v", aff, err)
		}
		for j, pt := range triangle {
			x, y := aff.Apply(pt.X, pt.Y)
			if math.Abs(x-v[j].X) > Zeroish || math.Abs(y-v[j].Y) > Zeroish {
				t.Errorf("[%d,%d] got=%f want=%f", i, j, Point{x, y}, v[j])
				t.Fatalf("aff=%f", aff)
			}
			rX, rY := inv.Apply(x, y)
			if math.Abs(pt.X-rX) > Zeroish || math.Abs(pt.Y-rY) > Zeroish {
				t.Fatalf("inverse of aff=%f inv=%v failed: %f != (%f,%f)", aff, inv, pt, rX, rY)
			}
		}
	}
}

func TestAffineXY(t *testing.T) {
	from := []Point{
		{-20.030589133069473, 0},
		{19.869795306245308, 0},
		{-20.4307123085543, 0},
		{19.619621722740685, 0},
		{-40.08048457364408, 0},
		{39.845042396654286, 0},
		{-40.38062805822299, 0},
		{39.64493670018861, 0},
		{0, 70.26702376778884},
		{0, 70.04202542947003},
		{0, -69.65861552647617},
		{0, -69.8835729634595},
		{0, 50.39198506924882},
		{0, 49.991571110506946},
		{0, -49.558665013911316},
		{0, -49.88442824953612},
	}
	to := []Point{
		{-19.8, -0.15},
		{20.1, -0.325},
		{-20.2, -0.125},
		{19.85, -0.25},
		{-39.85, -0.15},
		{40.075, -0.425},
		{-40.15, -0.1},
		{39.875, -0.4},
		{0.5, 70.025},
		{0.5, 69.8},
		{0.1, -69.9},
		{0.125, -70.125},
		{0.45, 50.15},
		{0.15, 49.75},
		{0.1, -49.8},
		{-0.075, -50.125},
	}
	aff, err := DeriveAffine(from, to)
	if err != nil {
		t.Fatalf("unable to derive affine: %v", err)
	}
	if wantAxx := 0.9999927663957388; math.Abs(wantAxx-aff.Axx) > Zeroish {
		t.Errorf("unexpected Axx: got=%f want=%f", aff.Axx, wantAxx)
	}
	if wantAxy := 0.0028083051938289495; math.Abs(wantAxy-aff.Axy) > Zeroish {
		t.Errorf("unexpected Axy: got=%f want=%f", aff.Axy, wantAxy)
	}
	if wantAyx := -0.0036293030188853086; math.Abs(wantAyx-aff.Ayx) > Zeroish {
		t.Errorf("unexpected Ayx: got=%f want=%f", aff.Ayx, wantAyx)
	}
	if wantAyy := 0.9999942766937774; math.Abs(wantAyy-aff.Ayy) > Zeroish {
		t.Errorf("unexpected Ayy: got=%f want=%f", aff.Ayy, wantAyy)
	}
	if wantDx := 0.23051307542745367; math.Abs(wantDx-aff.Dx) > Zeroish {
		t.Errorf("unexpected Dx got=%f want=%f", aff.Dx, wantDx)
	}
	if wantDy := -0.24152285331242462; math.Abs(wantDy-aff.Dy) > Zeroish {
		t.Errorf("unexpected Dy got=%f want=%f", aff.Dy, wantDy)
	}
}
