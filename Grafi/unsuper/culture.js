function load_culture() {
        Bokeh.safely(function() {
          (function(root) {
            function embed_document(root) {
var x = document.getElementsByClassName("main-root")[0];
x.setAttribute('id','f76da686-0933-4cfc-a45d-4f1f04c37a1d');
x.setAttribute('data-root-id', '9239');
render_items = [{"docid":"e5db6aa8-cc8c-4164-b514-e94a5cd9fecd","root_ids":["9239"],"roots":{"9239":"f76da686-0933-4cfc-a45d-4f1f04c37a1d"}}];
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