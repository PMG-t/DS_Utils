function load_graph_nightlife_link_1() {
        Bokeh.safely(function() {
          (function(root) {
            function embed_document(root) {
var x = document.getElementsByClassName("main-root")[0];
x.setAttribute('id','9e6bd1f3-f5da-4081-96f7-d3289b1047d2');
x.setAttribute('data-root-id', '14501');
render_items = [{"docid":"a582e94a-ba13-4470-b65e-0e4ab8902587","root_ids":["14501"],"roots":{"14501":"9e6bd1f3-f5da-4081-96f7-d3289b1047d2"}}];
root.Bokeh.embed.embed_items(docs_json, render_items);

            }
            if (root.Bokeh !== undefined) {
              embed_document(root);
            } else {
              var attempts = 0;
              var timer = setInterval(function(root) {
                if (root.Bokeh !== undefined) {
                  clearInterval(timer);
                  embed_document(root);
                } else {
                  attempts++;
                  if (attempts > 100) {
                    clearInterval(timer);
                    console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                  }
                }
              }, 10, root)
            }
          })(window);
        });
    };