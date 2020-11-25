### Visualization routines to aid development
# Written by Boris Bonev, Nov. 2020

## routine for plotting ranks
rectangle(w, h, x, y) = Shape([0,w,w,0] .+ x , [0,0,h,h] .+ y)

function _plotranks!(hssA::HssMatrix, ro, co)
  if hssA.leafnode
    m, n = size(hssA.D)
    plot!(rectangle(n,m,co,ro), color=:deepskyblue, label=false)
  else
    _plotranks!(hssA.A11, ro, co)
    _plotranks!(hssA.A22, ro+hssA.m1, co+hssA.n1)
    # plot off-diagonal blocks
    plot!(rectangle(hssA.m2,hssA.n1,co+hssA.m1,ro), color=:grey95, label=false)
    annotate!((co+hssA.m1 + 0.5*hssA.m2, ro + 0.5*hssA.n1, text(rank(hssA.B12), 8)))
    plot!(rectangle(hssA.m1,hssA.n2,co,ro+hssA.n1), color=:grey95, label=false)
    annotate!((co + 0.5*hssA.m1, ro+hssA.n1 + 0.5*hssA.n2, text(rank(hssA.B21), 8)))
  end
end

function plotranks(hssA::HssMatrix)
  plot(yflip=true)
  _plotranks!(hssA,0.5,0.5)
end

