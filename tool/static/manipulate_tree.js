function get_cluster() {
  Url = 'http://127.0.0.1:5000/get_cluster'
  var subject_system = document.getElementById('subject_system_id').value;


  $.getJSON(Url, {
    subject_system: subject_system
  }, function (result) {
    // alert(result);

    setupDiagram(result);

    console.log(result);
  })
  clearDiagram();

}



function init() {
  //if (window.goSamples) goSamples();  // init for these samples -- you don't need to call this
  var $ = go.GraphObject.make; // for conciseness in defining templates
  myFullDiagram =
    $(go.Diagram, "fullDiagram", // each diagram refers to its DIV HTML element by id
      {
        initialAutoScale: go.Diagram.UniformToFill, // automatically scale down to show whole tree
        //maxScale: 0.25,
        contentAlignment: go.Spot.Center, // center the tree in the viewport
        isReadOnly: false, // don't allow user to change the diagram
        "animationManager.isEnabled": true,
        // layout: $(go.TreeLayout,
        //   { angle: 90, sorting: go.TreeLayout.SortingAscending }),
        layout: $(FlatTreeLayout, // custom Layout, defined below
          {
            angle: 90,
            compaction: go.TreeLayout.CompactionNone
          }),
        maxSelectionCount: 1, // only one node may be selected at a time in each diagram
        // when the selection changes, update the myLocalDiagram view
        "undoManager.isEnabled": true
      });

  

  //myFullDiagram.toolManager.textEditingTool.defaultTextEditor = window.TextEditor;
  //myLocalDiagram.toolManager.textEditingTool.defaultTextEditor = window.TextEditor;

  // Define a node template that is shared by both diagrams


  var myNodeTemplate =
    $(go.Node, "Auto", {
        isTreeExpanded: false
      }, // by default collapsed
      {
        locationSpot: go.Spot.Center
      },
      new go.Binding("text", "key", go.Binding.toString), // for sorting
      $(go.Shape, "Rectangle",
        new go.Binding("fill", "color"), {
          stroke: null,
          name: "SHAPE"
        }),
      $(go.TextBlock, {
          margin: 5,
          editable: true
        },
        new go.Binding("text", "node_text", function (k) {
          return "" + k;
        })), {
        toolTip: // define a tooltip for each node
          $(go.Adornment, "Spot", // that has several labels around it
            {
              background: "transparent"
            }, // avoid hiding tooltip when mouse moves
            $(go.Placeholder, {
              padding: 5
            }),
            $(go.TextBlock, {
                alignment: go.Spot.Top,
                alignmentFocus: go.Spot.Bottom,
                stroke: "red"
              },
              new go.Binding("text", "key", function (s) {
                return "key: " + s;
              })),
            $(go.TextBlock, "Bottom", {
                alignment: go.Spot.Bottom,
                alignmentFocus: go.Spot.Top,
                stroke: "red"
              },
              new go.Binding("text", "node_text", function (s) {
                return "Cluster Name: " + s;
              }))
          ) // end Adornment
      },
      $("TreeExpanderButton")

    );
  myFullDiagram.nodeTemplate = myNodeTemplate;
  

  // Define a basic link template, not selectable, shared by both diagrams
  var myLinkTemplate =
    $(go.Link, {
        routing: go.Link.Normal,
        selectable: false
      },
      $(go.Shape, {
        strokeWidth: 1
      })
    );
  myFullDiagram.linkTemplate = myLinkTemplate;
  


  // Create the full tree diagram
  // setupDiagram();

  // Create a part in the background of the full diagram to highlight the selected node
  highlighter =
    $(go.Part, "Auto", {
        layerName: "Background",
        selectable: false,
        isInDocumentBounds: false,
        locationSpot: go.Spot.Center
      },
      $(go.Shape, "Ellipse", {
        fill: $(go.Brush, "Radial", {
          0.0: "yellow",
          1.0: "white"
        }),
        stroke: null,
        desiredSize: new go.Size(400, 400)
      })
    );
  myFullDiagram.add(highlighter);

  jQuery('#technique_choice_id').change(function () {
    var technique_choice = document.getElementById('technique_choice_id').value;

    myFullDiagram.nodes.each(function (n) {
      update_node_text(n, technique_choice);
    });

  });

  function update_node_text(node, technique) {
    myFullDiagram.model.commit(function (m) { // this Model
      // This is the safe way to change model data
      // GoJS will be notified that the data has changed
      // and can update the node in the Diagram
      // and record the change in the UndoManager
      if (technique == 'tfidf_word') {
        m.set(node.data, "node_text", node.data.tfidf_word);
      } else if (technique == 'tfidf_method') {
        m.set(node.data, "node_text", node.data.tfidf_method);
      } else if (technique == 'lda_word') {
        m.set(node.data, "node_text", node.data.lda_word);
      } else if (technique == 'lda_method') {
        m.set(node.data, "node_text", node.data.lda_method);
      } else if (technique == 'lsi_word') {
        m.set(node.data, "node_text", node.data.lsi_word);
      } else if (technique == 'lsi_method') {
        m.set(node.data, "node_text", node.data.lsi_method);
      }

    }, "update node text");
  }

  jQuery('#search_button').click(function(){
    alert('hello');
    change_node_color();
  });

  function change_node_color(){
    myFullDiagram.nodes.each(function (n) {
      myFullDiagram.model.commit(function (m) {
        m.set(n.data, "color", "#ffc61a" );
      }, 'change node color');
    });
  }

  myFullDiagram.addDiagramListener("ObjectContextClicked",
    function (e) {
      var part = e.subject.part;
      if (!(part instanceof go.Link)) {
        // showUserStudyPanel(part);
        showNodeDetails(part);
      }

    });



  // Start by focusing the diagrams on the node at the top of the tree.
  // Wait until the tree has been laid out before selecting the root node.
  myFullDiagram.addDiagramListener("InitialLayoutCompleted", function (e) {
    var node0 = myFullDiagram.findPartForKey(0);
    console.log(node0);
    if (node0 !== null) node0.isSelected = true;
    e.diagram.findTreeRoots().each(function (r) {
      r.expandTree(3);
    });

  });

  // myFullDiagram.toolManager.mouseMoveTools.insertAt(2, new DragZoomingTool());

}

function showNodeDetails(part){
    document.getElementById('node_key').innerHTML = 'Node Key: ' + part.data.key;
    document.getElementById('node_summary').innerHTML = part.data.text_summary;
    document.getElementById('node_patterns').innerHTML = part.data.spm_method;
    document.getElementById('files').innerHTML = part.data.files;
    document.getElementById('files_count').innerHTML = part.data.files_count;

    // jQuery('#node_details').modal('show');
}


// Customize the TreeLayout to position all of the leaf nodes at the same vertical Y position.
function FlatTreeLayout() {
  go.TreeLayout.call(this); // call base constructor
}
go.Diagram.inherit(FlatTreeLayout, go.TreeLayout);

// This assumes the TreeLayout.angle is 90 -- growing downward
FlatTreeLayout.prototype.commitLayout = function () {
  go.TreeLayout.prototype.commitLayout.call(this); // call base method first
  // find maximum Y position of all Nodes
  var y = -Infinity;
  this.network.vertexes.each(function (v) {
    y = Math.max(y, v.node.position.y);
  });
  // move down all leaf nodes to that Y position, but keeping their X position
  this.network.vertexes.each(function (v) {
    if (v.destinationEdges.count === 0) {
      // shift the node down to Y
      v.node.position = new go.Point(v.node.position.x, y);
      // extend the last segment vertically
      v.node.toEndSegmentLength = Math.abs(v.centerY - y);
    } else { // restore to normal value
      v.node.toEndSegmentLength = 10;
    }
  });
};
// end FlatTreeLayout


// Make the corresponding node in the full diagram to that selected in the local diagram selected,
// then call showLocalOnFullClick to update the local diagram.








function setupDiagram(result) {
  var nodeDataArray = [];
  for (x in result) {
    nodeDataArray.push({
      key: result[x].key,
      parent: result[x].parent,
      node_text: result[x].tfidf_word,
      tfidf_word: result[x].tfidf_word,
      tfidf_method: result[x].tfidf_method,
      lda_word: result[x].lda_word,
      lda_method: result[x].lda_method,
      lsi_word: result[x].lsi_word,
      lsi_method: result[x].lsi_method,
      color: "#cce6ff",
      spm_method: result[x].spm_method,
      text_summary: result[x].text_summary,
      files: result[x].files,
      files_count: result[x].files_count
    });

  }
  // Use below line for randomly coloring brushes
  // color: go.Brush.randomColor()

  myFullDiagram.model = new go.TreeModel(nodeDataArray);
  // update_nodes_for_study();

}


function clearDiagram() {
  myFullDiagram.model = null;
}




