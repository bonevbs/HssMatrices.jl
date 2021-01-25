### Visualization routines to aid development
# This entire implementation is extremely hacky...
# Ideally this should be eventually replaced by something based on RecipesBase
#
# Written by Boris Bonev, Nov. 2020

## routine for plotting ranks
rectangle(w, h, x, y) = Shape([0,w,w,0] .+ x , [0,0,h,h] .+ y)

# recursive plotting routine
function _plotranks!(hssA::HssMatrix, ro, co, rticks, cticks, level; cl=4)
  if hssA.leafnode
    m, n = size(hssA.D)
    plot!(rectangle(n,m,co,ro), color=:tomato, label=false)
  else
    _plotranks!(hssA.A11, ro, co, rticks, cticks, level+1)
    if level < cl
      append!(rticks, ro+hssA.m1)
      append!(cticks, co+hssA.n1)
    end
    _plotranks!(hssA.A22, ro+hssA.m1, co+hssA.n1, rticks, cticks, level+1)
    # plot off-diagonal blocks
    plot!(rectangle(hssA.m2,hssA.n1,co+hssA.m1,ro), color=:aliceblue, label=false)
    if level < cl
      annotate!((co+hssA.m1 + 0.5*hssA.m2, ro + 0.5*hssA.n1, text(rank(hssA.B12), 8)))
    end
    plot!(rectangle(hssA.m1,hssA.n2,co,ro+hssA.n1), color=:aliceblue, label=false)
    if level < cl
      annotate!((co + 0.5*hssA.m1, ro+hssA.n1 + 0.5*hssA.n2, text(rank(hssA.B21), 8)))
    end
  end
  return rticks, cticks
end

function plotranks(hssA::HssMatrix; cutoff_level=4)
  m,n = size(hssA)
  aspect = m/n
  plot(yflip=true, showaxis=false, size = (400, 400*aspect))
  xticks = [1]; yticks = [1]
  yticks, xticks = _plotranks!(hssA,1,1,yticks,xticks,0;cl=cutoff_level)
  append!(yticks, m+1)
  append!(xticks, n+1)
  plot!(aspect_ratio=:equal)
  plot_ref = plot!(xticks=xticks, yticks=yticks, xmirror=true)
  return plot_ref
end

# convenience function imitating the behavior of pcolor in matlab
pcolor(A) = heatmap(A, xlims=(1,size(A,2)), ylims=(1,size(A,1)), yflip=true, aspect_ratio=:equal);
