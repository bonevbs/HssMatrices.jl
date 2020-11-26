### Visualization routines to aid development
# This entire implementation is extremely hacky...
# Ideally this should be eventually replaced by something based on RecipesBase
#
# Written by Boris Bonev, Nov. 2020

## routine for plotting ranks
rectangle(w, h, x, y) = Shape([0,w,w,0] .+ x , [0,0,h,h] .+ y)

# recursive plotting routine
function _plotranks!(hssA::HssMatrix, ro, co, rticks, cticks)
  if hssA.leafnode
    m, n = size(hssA.D)
    plot!(rectangle(n,m,co,ro), color=:tomato, label=false)
  else
    _plotranks!(hssA.A11, ro, co, rticks, cticks)
    append!(rticks, ro+hssA.m1)
    append!(cticks, co+hssA.n1)
    _plotranks!(hssA.A22, ro+hssA.m1, co+hssA.n1, rticks, cticks)
    # plot off-diagonal blocks
    plot!(rectangle(hssA.m2,hssA.n1,co+hssA.m1,ro), color=:aliceblue, label=false)
    annotate!((co+hssA.m1 + 0.5*hssA.m2, ro + 0.5*hssA.n1, text(rank(hssA.B12), 8)))
    plot!(rectangle(hssA.m1,hssA.n2,co,ro+hssA.n1), color=:aliceblue, label=false)
    annotate!((co + 0.5*hssA.m1, ro+hssA.n1 + 0.5*hssA.n2, text(rank(hssA.B21), 8)))
  end
  return rticks, cticks
end

function plotranks(hssA::HssMatrix)
  m,n = size(hssA)
  aspect = m/n
  plot(yflip=true, showaxis=false, size = (400, 400*aspect))
  xticks = [1]; yticks = [1]
  yticks, xticks = _plotranks!(hssA,1,1,yticks,xticks)
  append!(yticks, m+1)
  append!(xticks, n+1)
  plot!(aspect_ratio=:equal)
  plot_ref = plot!(xticks=xticks, yticks=yticks, xmirror=true)
  return plot_ref
end

