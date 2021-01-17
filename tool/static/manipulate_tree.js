function get_cluster() {
    Url = 'http://127.0.0.1:5000/get_cluster'
    var subject_system = document.getElementById('subject_system_id').value;
  
    
    $.getJSON(Url, {subject_system : subject_system}, function (result) {
            // alert(result);

            setupDiagram(result);

            console.log(result);
        })
    clearDiagram();

}



 function init() {
  //if (window.goSamples) goSamples();  // init for these samples -- you don't need to call this
  var $ = go.GraphObject.make;  // for conciseness in defining templates
  myFullDiagram =
    $(go.Diagram, "fullDiagram",  // each diagram refers to its DIV HTML element by id
      {
        initialAutoScale: go.Diagram.UniformToFill,  // automatically scale down to show whole tree
        //maxScale: 0.25,
        contentAlignment: go.Spot.Center,  // center the tree in the viewport
        isReadOnly: false,  // don't allow user to change the diagram
        "animationManager.isEnabled": true,
        // layout: $(go.TreeLayout,
        //   { angle: 90, sorting: go.TreeLayout.SortingAscending }),
        layout:
          $(FlatTreeLayout,  // custom Layout, defined below
            {
              angle: 90,
              compaction: go.TreeLayout.CompactionNone
            }),
        maxSelectionCount: 1,  // only one node may be selected at a time in each diagram
        // when the selection changes, update the myLocalDiagram view
        "ChangedSelection": showLocalOnFullClick,
          "undoManager.isEnabled": true
      });

  myLocalDiagram =  // this is very similar to the full Diagram
    $(go.Diagram, "localDiagram",
      {
        autoScale: go.Diagram.UniformToFill,
        contentAlignment: go.Spot.Center,
        isReadOnly: false,
        layout: $(go.TreeLayout,
          { angle: 90, sorting: go.TreeLayout.SortingAscending }),
        "LayoutCompleted": function(e) {
          var sel = e.diagram.selection.first();
          if (sel !== null) myLocalDiagram.scrollToRect(sel.actualBounds);
        },
        maxSelectionCount: 1,
        // when the selection changes, update the contents of the myLocalDiagram
        "ChangedSelection": showLocalOnLocalClick,
          "undoManager.isEnabled": true
      });

  //myFullDiagram.toolManager.textEditingTool.defaultTextEditor = window.TextEditor;
  //myLocalDiagram.toolManager.textEditingTool.defaultTextEditor = window.TextEditor;

  // Define a node template that is shared by both diagrams


  var myNodeTemplate =
    $(go.Node, "Auto",
        {isTreeExpanded: false },  // by default collapsed
      { locationSpot: go.Spot.Center },
      new go.Binding("text", "key", go.Binding.toString),  // for sorting
      $(go.Shape, "Rectangle",
        new go.Binding("fill", "color"),
        { stroke: null, name: "SHAPE" }),
      $(go.TextBlock,
        { margin: 5, editable:true},
        new go.Binding("text", "node_text", function(k) { return "" + k ; })),
        {
            toolTip:                       // define a tooltip for each node
              $(go.Adornment, "Spot",      // that has several labels around it
                { background: "transparent" },  // avoid hiding tooltip when mouse moves
                $(go.Placeholder, { padding: 5 }),
                $(go.TextBlock,
                  { alignment: go.Spot.Top, alignmentFocus: go.Spot.Bottom, stroke: "red"},
                  new go.Binding("text", "key", function(s) { return "key: " + s; })),
                $(go.TextBlock, "Bottom",
                  { alignment: go.Spot.Bottom, alignmentFocus: go.Spot.Top, stroke: "red" },
                  new go.Binding("text", "node_text", function(s) { return "Cluster Name: " + s; }))
              )  // end Adornment
        },
        $("TreeExpanderButton")

    );
  myFullDiagram.nodeTemplate = myNodeTemplate;
  myLocalDiagram.nodeTemplate = myNodeTemplate;

  // Define a basic link template, not selectable, shared by both diagrams
  var myLinkTemplate =
    $(go.Link,
      { routing: go.Link.Normal, selectable: false },
      $(go.Shape,
        { strokeWidth: 1 })
    );
  myFullDiagram.linkTemplate = myLinkTemplate;
  myLocalDiagram.linkTemplate = myLinkTemplate;


  // Create the full tree diagram
  // setupDiagram();

  // Create a part in the background of the full diagram to highlight the selected node
  highlighter =
    $(go.Part, "Auto",
      {
        layerName: "Background",
        selectable: false,
        isInDocumentBounds: false,
        locationSpot: go.Spot.Center
      },
      $(go.Shape, "Ellipse",
        {
          fill: $(go.Brush, "Radial", { 0.0: "yellow", 1.0: "white" }),
          stroke: null,
          desiredSize: new go.Size(400, 400)
        })
    );
  myFullDiagram.add(highlighter);

  jQuery('#technique_choice_id').change(function(){
    var technique_choice = document.getElementById('technique_choice_id').value;

    myFullDiagram.nodes.each(function(n){
      update_node_text(n, technique_choice);
    });
    
  });

  function update_node_text(node, technique){
    myFullDiagram.model.commit(function(m) {  // this Model
      // This is the safe way to change model data
      // GoJS will be notified that the data has changed
      // and can update the node in the Diagram
      // and record the change in the UndoManager
      if (technique == 'tfidf_word') {
        m.set(node.data, "node_text", node.data.tfidf_word);
      } else if (technique == 'tfidf_method'){
        m.set(node.data, "node_text", node.data.tfidf_method);
      } else if (technique == 'lda_word'){
        m.set(node.data, "node_text", node.data.lda_word);
      } else if (technique == 'lda_method'){
        m.set(node.data, "node_text", node.data.lda_method);
      } else if (technique == 'lsi_word'){
        m.set(node.data, "node_text", node.data.lsi_word);
      } else if (technique == 'lsi_method'){
        m.set(node.data, "node_text", node.data.lsi_method);
      }

    }, "update node text");
  }

  myFullDiagram.addDiagramListener("ObjectContextClicked",
  function(e) {
    var part = e.subject.part;
    if (!(part instanceof go.Link))
    {
        showModal(part);
    }

  });



  // Start by focusing the diagrams on the node at the top of the tree.
  // Wait until the tree has been laid out before selecting the root node.
  myFullDiagram.addDiagramListener("InitialLayoutCompleted", function(e) {
    var node0 = myFullDiagram.findPartForKey(0);
    console.log(node0);
    if (node0 !== null) node0.isSelected = true;
    showLocalOnFullClick();
    e.diagram.findTreeRoots().each(function(r) { r.expandTree(3); });

  });

  myFullDiagram.toolManager.mouseMoveTools.insertAt(2, new DragZoomingTool());

}


// Customize the TreeLayout to position all of the leaf nodes at the same vertical Y position.
function FlatTreeLayout() {
  go.TreeLayout.call(this);  // call base constructor
}
go.Diagram.inherit(FlatTreeLayout, go.TreeLayout);

// This assumes the TreeLayout.angle is 90 -- growing downward
FlatTreeLayout.prototype.commitLayout = function() {
  go.TreeLayout.prototype.commitLayout.call(this);  // call base method first
  // find maximum Y position of all Nodes
  var y = -Infinity;
  this.network.vertexes.each(function(v) {
    y = Math.max(y, v.node.position.y);
  });
  // move down all leaf nodes to that Y position, but keeping their X position
  this.network.vertexes.each(function(v) {
    if (v.destinationEdges.count === 0) {
      // shift the node down to Y
      v.node.position = new go.Point(v.node.position.x, y);
      // extend the last segment vertically
      v.node.toEndSegmentLength = Math.abs(v.centerY - y);
    } else {  // restore to normal value
      v.node.toEndSegmentLength = 10;
    }
  });
};
// end FlatTreeLayout


// Make the corresponding node in the full diagram to that selected in the local diagram selected,
// then call showLocalOnFullClick to update the local diagram.
function showLocalOnLocalClick() {
  var selectedLocal = myLocalDiagram.selection.first();
  if (selectedLocal !== null) {
    // there are two separate Nodes, one for each Diagram, but they share the same key value
    myFullDiagram.select(myFullDiagram.findPartForKey(selectedLocal.data.key));
  }
}



function showLocalOnFullClick() {
  var node = myFullDiagram.selection.first();
  if (node !== null) {
    // make sure the selected node is in the viewport
    myFullDiagram.scrollToRect(node.actualBounds);
    // move the large yellow node behind the selected node to highlight it
    highlighter.location = node.location;
    // create a new model for the local Diagram
    var model = new go.TreeModel();
    // add the selected node and its children and grandchildren to the local diagram
    var nearby = node.findTreeParts(3);  // three levels of the (sub)tree
    // add parent and grandparent
    var parent = node.findTreeParentNode();
    if (parent !== null) {
      nearby.add(parent);
      var grandparent = parent.findTreeParentNode();
      if (grandparent !== null) {
        nearby.add(grandparent);
      }
    }
    // create the model using the same node data as in myFullDiagram's model
    nearby.each(function(n) {
      if (n instanceof go.Node) model.addNodeData(n.data);
    });
    myLocalDiagram.model = model;
    // select the node at the diagram's focus
    var selectedLocal = myLocalDiagram.findPartForKey(node.data.key);
    if (selectedLocal !== null) selectedLocal.isSelected = true;
  }
}





  

// Create the tree model containing TOTAL nodes, with each node having a variable number of children
// function setupDiagram(total) {
//   if (total === undefined) total = 100;  // default to 100 nodes
//   var nodeDataArray = [];
//   for (var i = 0; i < total; i++) {
//     nodeDataArray.push({
//       key: nodeDataArray.length,
//       color: go.Brush.randomColor()
//     });
//   }
//   var j = 0;
//   for (var i = 1; i < total; i++) {
//     var data = nodeDataArray[i];
//     data.parent = j;
//     if (Math.random() < 0.3) j++;  // this controls the likelihood that there are enough children
//   }
//   myFullDiagram.model = new go.TreeModel(nodeDataArray);
// }



function setupDiagram(result) {
    var nodeDataArray = [];
    for (x in result)
        {
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
                description: 'docstring : ' + result[x].text_summary+ '\n pattern: '+ result[x].spm_method,
                spm_method: result[x].spm_method,
                text_summary: result[x].text_summary
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


 function get_image() {
    myFullDiagram.makeImage();
    alert('Image saved');
}

function printDiagram() {
  var svgWindow = window.open();
  if (!svgWindow) return;  // failure to open a new Window
  var printSize = new go.Size(700, 960);
  var bnds = myFullDiagram.documentBounds;
  var x = bnds.x;
  var y = bnds.y;
  while (y < bnds.bottom) {
    while (x < bnds.right) {
      var svg = myFullDiagram.makeSVG({ scale: 1.0, position: new go.Point(x, y), size: printSize });
      svgWindow.document.body.appendChild(svg);
      x += printSize.width;
    }
    x = bnds.x;
    y += printSize.height;
  }
  setTimeout(function() { svgWindow.print(); }, 1);
}
